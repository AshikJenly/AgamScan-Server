"""
NER service using Azure AI for field extraction
Extracts structured fields from OCR text using LLM
"""

import json
import re
from typing import Dict, Tuple, List, Any
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

from config import (
    AZURE_AI_ENDPOINT,
    AZURE_AI_KEY,
    AZURE_AI_MODEL_NAME,
    AZURE_API_VERSION,
    REQUIRED_FIELDS
)


class NERService:
    """Service for Named Entity Recognition using Azure AI"""
    
    def __init__(
        self,
        endpoint: str = AZURE_AI_ENDPOINT,
        key: str = AZURE_AI_KEY,
        model_name: str = AZURE_AI_MODEL_NAME,
        api_version: str = AZURE_API_VERSION
    ):
        """
        Initialize NER service
        
        Args:
            endpoint: Azure AI endpoint
            key: Azure AI API key
            model_name: Model name to use
            api_version: API version
        """
        self.endpoint = endpoint
        self.key = key
        self.model_name = model_name
        self.api_version = api_version
        self.client = None
    
    def initialize(self) -> bool:
        """Initialize Azure AI client"""
        try:
            if not self.endpoint or not self.key:
                print("❌ Azure AI credentials not configured")
                return False
            
            self.client = ChatCompletionsClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.key),
                api_version=self.api_version
            )
            print("✅ NER service initialized")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize NER service: {e}")
            return False
    
    def is_initialized(self) -> bool:
        """Check if service is initialized"""
        return self.client is not None
    
    def build_prompt(self, ocr_text: str) -> str:
        """
        Build universal prompt for global driving licence field extraction
        """
        return f"""
    You are an expert at interpreting OCR text from driver's licences and identity cards
    from any country (e.g., India, USA, Australia, China, UK, Canada, etc.).
    You must extract structured fields accurately into a consistent JSON format.

    OCR TEXT:
    \"\"\"{ocr_text}\"\"\"

    OUTPUT REQUIREMENTS:
    - Return only a valid JSON object, strictly matching this schema:
    {json.dumps(REQUIRED_FIELDS, indent=4)}

    ### Rules:
    1. Use only the visible OCR text — never invent or assume unseen data.
    2. Accurately detect which country or region the licence belongs to.
    3. "Issuing Authority" → Use the region or state mentioned (e.g., "California", "Tamil Nadu", "New South Wales", "Beijing").
    4. "First Name" and "Last Name" → Based on the person’s name on the licence. 
    - For initials or single names, map intelligently (e.g., "C. THIRUNAVUKARASU" → First: THIRUNAVUKARASU, Last: C).
    5. "Licence Number" → The unique alphanumeric ID of the licence.
    6. "Date Of Birth" and "Date Of Expiry" → Keep date format as shown (dd/mm/yyyy, yyyy-mm-dd, etc.).
    7. "Gender" → M, F, X, Male, Female, or blank if not shown.
    8. "Address" → Merge all address lines neatly, comma-separated, without extra newlines.
    9. "Country" → Detect automatically (e.g., “India”, “Australia”, “United States”, “China”, etc.).
    10. Leave any field blank ("") if not visible.
    11. Output must be **only raw JSON**, with no markdown, code fences, or commentary.

    ### Example Output:
    {{
    "Issuing Authority": "New South Wales",
    "First Name": "ALEXANDER",
    "Last Name": "SMITH",
    "Licence Number": "123456789",
    "Date Of Birth": "15/03/1987",
    "Date Of Expiry": "15/03/2027",
    "Gender": "Male",
    "Address": "24 KING STREET, SYDNEY NSW 2000",
    "Country": "Australia"
    }}
    """

    def safe_json_parse(self, raw_text: str) -> Dict[str, str]:
        """
        Clean and safely parse JSON from model output
        
        Args:
            raw_text: Raw model output
            
        Returns:
            Parsed JSON or default empty fields
        """
        raw = raw_text.strip()
        
        # Remove code fences like ```json ... ```
        raw = re.sub(r"^```(?:json)?", "", raw)
        raw = re.sub(r"```$", "", raw)
        raw = raw.strip()
        
        # Try to find first and last braces
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end+1]
        
        try:
            return json.loads(raw)
        except Exception as e:
            print(f"⚠️ JSON parsing failed: {e}")
            return {k: "" for k in REQUIRED_FIELDS}
    
    def compute_field_confidence(
        self,
        field_value: str,
        ocr_words: List[Dict[str, Any]]
    ) -> float:
        """
        Compute confidence for extracted field based on OCR word confidences
        
        Args:
            field_value: Extracted field value
            ocr_words: List of OCR words with confidence scores
            
        Returns:
            Average confidence score
        """
        if not field_value:
            return 0.0
        
        # Tokenize field value
        tokens = re.findall(r'\w+', field_value.lower())
        
        # Find matching OCR words
        matched_confidences = []
        for token in tokens:
            for word in ocr_words:
                if word["text"].lower() == token:
                    matched_confidences.append(word["confidence"])
                    break
        
        if not matched_confidences:
            return 0.0
        
        return round(sum(matched_confidences) / len(matched_confidences), 3)
    
    def extract_fields(
        self,
        ocr_data: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Dict[str, Any]], str]:
        """
        Extract structured fields from OCR data
        
        Args:
            ocr_data: OCR data from OCR service
            
        Returns:
            Tuple of (success, extracted_fields, message)
            extracted_fields format:
            {
                "field_name": {
                    "value": "extracted_value",
                    "confidence": 0.95
                }
            }
        """
        if not self.is_initialized():
            return False, {}, "❌ NER service not initialized"
        
        try:
            # Get OCR text
            ocr_text = ocr_data.get("full_text", "")
            if not ocr_text:
                return False, {}, "❌ No OCR text provided"
            
            # Build prompt
            prompt = self.build_prompt(ocr_text)
            
            # Call LLM
            response = self.client.complete(
                model=self.model_name,
                messages=[
                    SystemMessage(
                        content="You extract structured details from OCR text into clean JSON."
                    ),
                    UserMessage(content=prompt)
                ],
                temperature=0
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            print(f"\n=== LLM Response ===\n{content[:500]}...")  # Log first 500 chars
            
            extracted = self.safe_json_parse(content)
            
            # Compute confidences for each field
            ocr_words = ocr_data.get("words", [])
            result = {}
            
            for field_name in REQUIRED_FIELDS.keys():
                value = extracted.get(field_name, "")
                confidence = self.compute_field_confidence(value, ocr_words)
                
                result[field_name] = {
                    "value": value,
                    "confidence": confidence
                }
            
            # Count non-empty fields
            filled_fields = sum(1 for f in result.values() if f["value"])
            
            return True, result, f"✅ Extracted {filled_fields}/{len(REQUIRED_FIELDS)} fields"
            
        except Exception as e:
            return False, {}, f"❌ Field extraction failed: {str(e)}"


if __name__ == "__main__":
    # Test NER service
    service = NERService()
    
    if service.initialize():
        print("\n✅ NER service initialized successfully")
        
        # Test with sample OCR data
        sample_ocr = {
            "full_text": "DRIVER LICENSE\nJOHN DOE\nLicense: D1234567\nDOB: 01/01/1990\nExpiry: 01/01/2030",
            "words": [
                {"text": "DRIVER", "confidence": 0.99},
                {"text": "LICENSE", "confidence": 0.99},
                {"text": "JOHN", "confidence": 0.98},
                {"text": "DOE", "confidence": 0.97},
            ]
        }
        
        print("\nTesting field extraction...")
        success, fields, message = service.extract_fields(sample_ocr)
        print(f"  {message}")
        
        if success:
            for field_name, field_data in fields.items():
                if field_data["value"]:
                    print(f"  {field_name}: {field_data['value']} (conf: {field_data['confidence']:.2f})")
    else:
        print("\n❌ Failed to initialize NER service")
