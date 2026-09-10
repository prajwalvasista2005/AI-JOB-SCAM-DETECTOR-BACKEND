import unittest
from datetime import datetime, timezone
from jose import jwt

from app.core.config import settings
from app.core.security import DUMMY_HASH, hash_password, normalize_email, verify_password
from app.core.auth import create_access_token
from app.models import User, Scan
from app.schemas.auth import Token
from app.schemas.scan import InputType, ScanRequest, ScanResponse
from app.schemas.user import UserCreate, UserResponse
from app.services.scan_services import (
    calculate_risk_score,
    detect_input_type,
    get_verdict,
)
from app.main import app


class TestSecurity(unittest.TestCase):
    def test_password_hashing_and_verification(self):
        password = "SecurePassword123!"
        hashed = hash_password(password)
        self.assertNotEqual(password, hashed)
        self.assertTrue(verify_password(password, hashed))
        self.assertFalse(verify_password("WrongPassword!", hashed))

    def test_dummy_hash_for_timing_attacks(self):
        # Verifies that dummy hash is valid bcrypt and returns False
        result = verify_password("DummyTestPassword123", DUMMY_HASH)
        self.assertFalse(result)

    def test_normalize_email(self):
        self.assertEqual(normalize_email("  Test@Example.COM  "), "test@example.com")


class TestAuth(unittest.TestCase):
    def test_jwt_token_creation_and_payload(self):
        user_id = 42
        token = create_access_token(user_id)
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        self.assertEqual(payload.get("sub"), str(user_id))
        self.assertIn("exp", payload)


class TestSchemas(unittest.TestCase):
    def test_scan_response_from_orm(self):
        now = datetime.now(timezone.utc)
        scan = Scan(
            id=1,
            user_id=10,
            input_type="text",
            content="Urgent hiring for data entry role whatsapp only",
            risk_score=25,
            verdict="Low Risk",
            created_at=now,
        )
        response_model = ScanResponse.model_validate(scan)
        self.assertEqual(response_model.id, 1)
        self.assertEqual(response_model.input_type, InputType.TEXT)
        self.assertEqual(response_model.risk_score, 25)
        self.assertEqual(response_model.verdict, "Low Risk")

    def test_user_response_from_orm(self):
        now = datetime.now(timezone.utc)
        user = User(
            id=5,
            email="candidate@example.com",
            password_hash="hashed_pw",
            created_at=now,
        )
        response_model = UserResponse.model_validate(user)
        self.assertEqual(response_model.id, 5)
        self.assertEqual(response_model.email, "candidate@example.com")


class TestScanLogic(unittest.TestCase):
    def test_detect_input_type(self):
        self.assertEqual(detect_input_type("https://linkedin.com/jobs/123"), InputType.URL)
        self.assertEqual(detect_input_type("http://example.org/apply"), InputType.URL)
        self.assertEqual(detect_input_type("screenshot.png"), InputType.IMAGE)
        self.assertEqual(detect_input_type("document.jpg"), InputType.IMAGE)
        self.assertEqual(detect_input_type("Job offer text content"), InputType.TEXT)

    def test_calculate_risk_score(self):
        # "registration fee": 30, "guaranteed job": 25, "whatsapp only": 15 -> 70
        content = "We have a guaranteed job. Please pay the registration fee and contact us via whatsapp only."
        score = calculate_risk_score(content)
        self.assertEqual(score, 70)

    def test_calculate_risk_score_capped_at_100(self):
        # All scam patterns together exceed 100 -> should cap at 100
        content = "registration fee guaranteed job no interview whatsapp only urgent hiring"
        score = calculate_risk_score(content)
        self.assertEqual(score, 100)

    def test_verdicts(self):
        self.assertEqual(get_verdict(20), "Low Risk")
        self.assertEqual(get_verdict(30), "Low Risk")
        self.assertEqual(get_verdict(35), "Medium Risk")
        self.assertEqual(get_verdict(60), "Medium Risk")
        self.assertEqual(get_verdict(75), "High Risk")


class TestAppRoutes(unittest.TestCase):
    def test_routes_registered(self):
        paths = app.openapi()["paths"]
        self.assertIn("/health", paths)
        self.assertIn("/auth/login", paths)
        self.assertIn("/users/", paths)
        self.assertIn("/users/me", paths)
        self.assertIn("/scan/", paths)
        self.assertIn("/scan/{scan_id}", paths)


if __name__ == "__main__":
    unittest.main()
