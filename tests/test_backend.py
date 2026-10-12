import unittest
from datetime import datetime, timezone, timedelta
from jose import jwt
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi import HTTPException, Response

from app.core.config import settings
from app.core.security import DUMMY_HASH, hash_password, normalize_email, verify_password
from app.core.auth import create_access_token
from app.core.database import Base
from app.core.dependencies import get_current_user
from app.models import User, Scan
from app.schemas.auth import Token
from app.schemas.scan import InputType, ScanRequest, ScanResponse
from app.schemas.user import UserCreate, UserResponse
from app.services.scan_services import (
    calculate_risk_score,
    detect_input_type,
    get_verdict,
    create_scan,
    get_user_scans,
    get_scan_by_id,
    delete_scan,
)
from app.services.user_services import create_user, authenticate_user
from app.routers.scan import delete_scan_endpoint
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

    def test_token_schema(self):
        token = Token(access_token="fake.jwt.token", token_type="bearer")
        self.assertEqual(token.access_token, "fake.jwt.token")
        self.assertEqual(token.token_type, "bearer")

    def test_user_create_validation(self):
        valid = UserCreate(email="user@example.com", password="Password123")
        self.assertEqual(valid.email, "user@example.com")

        with self.assertRaises(ValidationError):
            UserCreate(email="not-an-email", password="Password123")

        with self.assertRaises(ValidationError):
            UserCreate(email="user@example.com", password="short")

        with self.assertRaises(ValidationError):
            UserCreate(email="user@example.com", password="a" * 73)

    def test_scan_request_validation(self):
        valid = ScanRequest(content="A valid job offer description with sufficient length.")
        self.assertIn("job offer", valid.content)

        # Content shorter than 10 chars
        with self.assertRaises(ValidationError):
            ScanRequest(content="Too short")

        # Whitespace-only content must be rejected
        with self.assertRaises(ValidationError):
            ScanRequest(content="               ")

        # Fewer than 10 non-whitespace characters must be rejected
        with self.assertRaises(ValidationError):
            ScanRequest(content="          ab")

        # Internal whitespace padding (e.g. "a        b") must not bypass minimum non-whitespace requirement
        with self.assertRaises(ValidationError):
            ScanRequest(content="a        b")


class TestScanLogic(unittest.TestCase):
    def test_detect_input_type(self):
        self.assertEqual(detect_input_type("https://linkedin.com/jobs/123"), InputType.URL)
        self.assertEqual(detect_input_type("http://example.org/apply"), InputType.URL)
        self.assertEqual(detect_input_type("screenshot.png"), InputType.IMAGE)
        self.assertEqual(detect_input_type("document.jpg"), InputType.IMAGE)
        self.assertEqual(detect_input_type("Job offer text content"), InputType.TEXT)

    def test_detect_input_type_edge_cases(self):
        # Multiline job description starting with a URL should be classified as TEXT
        multiline_url = "https://example.com/job\n\nWe are looking for a Senior Developer with 5 years experience."
        self.assertEqual(detect_input_type(multiline_url), InputType.TEXT)

        # URL followed by inline text should be classified as TEXT
        inline_url_text = "https://example.com and further job details"
        self.assertEqual(detect_input_type(inline_url_text), InputType.TEXT)

        # Text mentioning or ending in an image file name should be classified as TEXT
        text_ending_in_image = "We are hiring. Please see the requirements in attached screenshot.png"
        self.assertEqual(detect_input_type(text_ending_in_image), InputType.TEXT)

        # Whitespace only or empty should be TEXT
        self.assertEqual(detect_input_type(""), InputType.TEXT)
        self.assertEqual(detect_input_type("   "), InputType.TEXT)

        # Standalone URLs with surrounding whitespace should still be classified as URL
        self.assertEqual(detect_input_type("  https://example.com/job  "), InputType.URL)

        # Malformed URLs lacking hostname or scheme should be classified as TEXT
        self.assertEqual(detect_input_type("https://"), InputType.TEXT)
        self.assertEqual(detect_input_type("http://"), InputType.TEXT)
        self.assertEqual(detect_input_type("https:///path"), InputType.TEXT)

        # Additional image file formats and filenames with spaces
        self.assertEqual(detect_input_type("job ad.png"), InputType.IMAGE)
        self.assertEqual(detect_input_type("document.jpeg"), InputType.IMAGE)
        self.assertEqual(detect_input_type("chart.webp"), InputType.IMAGE)

    def test_calculate_risk_score(self):
        # "guaranteed job": 25, "whatsapp only": 15 -> 40
        content = "We have a guaranteed job. Contact us via whatsapp only."
        score = calculate_risk_score(content)
        self.assertEqual(score, 40)

        # "registration fee": 60 alone
        fee_content = "Please pay the registration fee before starting."
        fee_score = calculate_risk_score(fee_content)
        self.assertEqual(fee_score, 60)

    def test_calculate_risk_score_capped_at_100(self):
        # All scam patterns together exceed 100 -> should cap at 100
        content = "registration fee guaranteed job no interview whatsapp only urgent hiring"
        score = calculate_risk_score(content)
        self.assertEqual(score, 100)

    def test_registration_fee_yields_high_risk(self):
        # Explicit demand for registration fee must categorize as High Risk (AUD-06 resolution)
        content = "Pay a registration fee to secure your job placement."
        score = calculate_risk_score(content)
        verdict = get_verdict(score)
        self.assertEqual(score, 60)
        self.assertEqual(verdict, "High Risk")

    def test_calculate_risk_score_negation_handling(self):
        # Legitimate postings disclaiming fees should not trigger false positive
        no_fee = "Legitimate Company Inc. Note: There is NO registration fee required for applying."
        self.assertEqual(calculate_risk_score(no_fee), 0)

        never_fee = "ACME Corp will never ask for any registration fees."
        self.assertEqual(calculate_risk_score(never_fee), 0)

        without_fee = "We offer employment without any registration fee."
        self.assertEqual(calculate_risk_score(without_fee), 0)

        not_guaranteed = "This is a competitive opening, not a guaranteed job."
        self.assertEqual(calculate_risk_score(not_guaranteed), 0)

        # Mixed: negated mention and unnegated mention should still score the unnegated flag
        mixed = "No registration fee for testing, but a registration fee is required for materials."
        self.assertEqual(calculate_risk_score(mixed), 60)

        # Cross-sentence boundaries should not negate: 'No.' in sentence 1 does not negate fee in sentence 2
        cross_sentence = "No. A registration fee is required before orientation."
        self.assertEqual(calculate_risk_score(cross_sentence), 60)

    def test_verdicts(self):
        # Low Risk: 0 to 20
        self.assertEqual(get_verdict(0), "Low Risk")
        self.assertEqual(get_verdict(10), "Low Risk")
        self.assertEqual(get_verdict(20), "Low Risk")

        # Medium Risk: 21 to 50
        self.assertEqual(get_verdict(21), "Medium Risk")
        self.assertEqual(get_verdict(35), "Medium Risk")
        self.assertEqual(get_verdict(50), "Medium Risk")

        # High Risk: 51 to 100
        self.assertEqual(get_verdict(51), "High Risk")
        self.assertEqual(get_verdict(60), "High Risk")
        self.assertEqual(get_verdict(75), "High Risk")
        self.assertEqual(get_verdict(100), "High Risk")


class TestAppRoutes(unittest.TestCase):
    def test_routes_registered(self):
        paths = app.openapi()["paths"]
        self.assertIn("/health", paths)
        self.assertIn("/auth/login", paths)
        self.assertIn("/users/", paths)
        self.assertIn("/users/me", paths)
        self.assertIn("/scan/", paths)
        self.assertIn("/scan/{scan_id}", paths)


class TestDatabaseServices(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()

    def test_user_services_flow(self):
        user_dto = UserCreate(email="alice@example.com", password="Password123")
        user = create_user(user_dto, self.db)
        self.assertIsNotNone(user)
        self.assertEqual(user.email, "alice@example.com")

        # Duplicate email returns None
        dup = create_user(user_dto, self.db)
        self.assertIsNone(dup)

        # Authenticate success
        auth_user = authenticate_user("alice@example.com", "Password123", self.db)
        self.assertIsNotNone(auth_user)
        self.assertEqual(auth_user.id, user.id)

        # Wrong password returns None
        bad_pw = authenticate_user("alice@example.com", "WrongPassword", self.db)
        self.assertIsNone(bad_pw)

        # Nonexistent email triggers DUMMY_HASH timing mitigation and returns None
        non_user = authenticate_user("nobody@example.com", "Password123", self.db)
        self.assertIsNone(non_user)

    def test_scan_services_and_idor(self):
        u1 = create_user(UserCreate(email="u1@example.com", password="Password123"), self.db)
        u2 = create_user(UserCreate(email="u2@example.com", password="Password123"), self.db)

        scan1 = create_scan("Job offer urgent hiring whatsapp only guaranteed job", u1.id, self.db)
        scan2 = create_scan("Another job description with no interview guaranteed job", u2.id, self.db)

        # User 1 scans listing
        u1_scans = get_user_scans(u1.id, self.db)
        self.assertEqual(len(u1_scans), 1)
        self.assertEqual(u1_scans[0].id, scan1.id)

        # User 1 can fetch their own scan
        fetched = get_scan_by_id(scan1.id, u1.id, self.db)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.id, scan1.id)

        # IDOR check: User 2 cannot fetch User 1's scan
        idor_scan = get_scan_by_id(scan1.id, u2.id, self.db)
        self.assertIsNone(idor_scan)

        # IDOR check: User 2 cannot delete User 1's scan
        idor_delete = delete_scan(scan1.id, u2.id, self.db)
        self.assertFalse(idor_delete)

        # User 1 can delete their own scan
        own_delete = delete_scan(scan1.id, u1.id, self.db)
        self.assertTrue(own_delete)
        self.assertIsNone(get_scan_by_id(scan1.id, u1.id, self.db))

    def test_database_error_rollbacks(self):
        from unittest.mock import MagicMock

        # create_user rolls back on unhandled DB error
        mock_db = MagicMock()
        mock_db.execute.return_value.scalar_one_or_none.return_value = None
        mock_db.commit.side_effect = Exception("DB failure")
        with self.assertRaises(Exception):
            create_user(UserCreate(email="fail@example.com", password="Password123"), mock_db)
        mock_db.rollback.assert_called_once()

        # create_scan rolls back on unhandled DB error
        mock_scan_db = MagicMock()
        mock_scan_db.commit.side_effect = Exception("DB failure")
        with self.assertRaises(Exception):
            create_scan("Valid job description for testing", 1, mock_scan_db)
        mock_scan_db.rollback.assert_called_once()

        # delete_scan rolls back on unhandled DB error
        mock_del_db = MagicMock()
        mock_del_db.scalars.return_value.first.return_value = MagicMock()
        mock_del_db.commit.side_effect = Exception("DB failure")
        with self.assertRaises(Exception):
            delete_scan(1, 1, mock_del_db)
        mock_del_db.rollback.assert_called_once()


class TestDependencies(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()
        self.user = create_user(UserCreate(email="user@test.com", password="Password123"), self.db)

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()

    def test_get_current_user_valid_token(self):
        token = create_access_token(self.user.id)
        current = get_current_user(token=token, db=self.db)
        self.assertEqual(current.id, self.user.id)
        self.assertEqual(current.email, "user@test.com")

    def test_get_current_user_invalid_token(self):
        with self.assertRaises(HTTPException) as ctx:
            get_current_user(token="invalid.jwt.token", db=self.db)
        self.assertEqual(ctx.exception.status_code, 401)

    def test_get_current_user_expired_token(self):
        expire = datetime.now(timezone.utc) - timedelta(minutes=5)
        expired_token = jwt.encode(
            {"sub": str(self.user.id), "exp": expire},
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM,
        )
        with self.assertRaises(HTTPException) as ctx:
            get_current_user(token=expired_token, db=self.db)
        self.assertEqual(ctx.exception.status_code, 401)

    def test_get_current_user_missing_sub(self):
        token_no_sub = jwt.encode(
            {"exp": datetime.now(timezone.utc) + timedelta(minutes=5)},
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM,
        )
        with self.assertRaises(HTTPException) as ctx:
            get_current_user(token=token_no_sub, db=self.db)
        self.assertEqual(ctx.exception.status_code, 401)

    def test_get_current_user_invalid_sub(self):
        token_bad_sub = jwt.encode(
            {"sub": "not_an_int", "exp": datetime.now(timezone.utc) + timedelta(minutes=5)},
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM,
        )
        with self.assertRaises(HTTPException) as ctx:
            get_current_user(token=token_bad_sub, db=self.db)
        self.assertEqual(ctx.exception.status_code, 401)

    def test_get_current_user_nonexistent_user(self):
        token_unknown = create_access_token(99999)
        with self.assertRaises(HTTPException) as ctx:
            get_current_user(token=token_unknown, db=self.db)
        self.assertEqual(ctx.exception.status_code, 401)


class TestDeleteScanEndpoint(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()
        self.user = create_user(UserCreate(email="owner@test.com", password="Password123"), self.db)

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()

    def test_delete_scan_endpoint_success(self):
        scan = create_scan("Job offer with urgent hiring and no interview required.", self.user.id, self.db)
        res = delete_scan_endpoint(scan_id=scan.id, db=self.db, current_user=self.user)
        self.assertIsInstance(res, Response)
        self.assertEqual(res.status_code, 204)

    def test_delete_scan_endpoint_not_found(self):
        with self.assertRaises(HTTPException) as ctx:
            delete_scan_endpoint(scan_id=9999, db=self.db, current_user=self.user)
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
