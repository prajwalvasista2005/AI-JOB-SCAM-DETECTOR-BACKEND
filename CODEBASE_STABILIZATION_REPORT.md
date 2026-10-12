# AI Job Scam Detector Backend — Codebase Stabilization & Bug Fixing Report

**Report Date:** October 12, 2026  
**Auditor:** Senior Backend & Application Security Engineer  
**Target Repository:** `AI-JOB-SCAM-DETECTOR`  
**Current Branch:** `main`  
**Runtime Environment:** Windows 11 / Python 3.14.7 / Virtual Environment (`.\venv`)  

---

## 1. Executive Summary

This report documents the comprehensive audit, verification, and correction pass performed on the `AI-JOB-SCAM-DETECTOR` backend. The primary objective was to detect and fix existing defects, correct flawed heuristic scoring and verdict thresholds, eliminate confirmed dead code, resolve protocol and validation inconsistencies, and reinforce system reliability **without altering intended functionality, introducing new features, or redesigning the established Router → Service → SQLAlchemy → PostgreSQL architecture**.

### Key Outcomes:
1. **Critical Scoring Defect Resolved (AUD-06):** Corrected the scoring and verdict policy. An upfront `"registration fee"` is now assigned 60 points (up from 30) and correctly evaluates to **"High Risk"** (previously misclassified as "Low Risk"). Documented a clear 3-tier verdict policy (Low: 0–20, Medium: 21–50, High: 51–100) with regression tests for boundary conditions and the registration-fee scenario.
2. **False Positive Negation Handling in Scoring:** Implemented clause-aware negation and word-boundary matching for scam patterns. Benign postings with fraud warnings (e.g., *"We never ask for any registration fees"* or *"There is NO registration fee required to apply"*) no longer trigger false High Risk verdicts, while unnegated or mixed occurrences correctly trigger the risk flag.
3. **Input Type Classification Bug Resolved:** Fixed false-positive classifications in `detect_input_type` where multiline job postings starting with `http` or text mentioning `.png` were misidentified as URLs or Images. Verified against standalone URLs, multiline URLs, image filenames with spaces, text mentioning images, malformed URLs, and whitespace.
4. **Input Validation Hardened:** Enforced non-whitespace validation in `ScanRequest` by counting total non-whitespace characters (`sum(1 for c in v if not c.isspace()) >= 10`). This blocks both blank/whitespace strings and internally padded strings (e.g., `"a        b"`) from bypassing length requirements and polluting the database with empty scans.
5. **HTTP 204 Specification Compliance Restored:** Corrected `DELETE /scan/{scan_id}` to declare `response_class=Response` and return an explicit `Response(status_code=204)`, eliminating an illegal `Content-Type: application/json` header and invalid OpenAPI response body schema on HTTP 204 No Content responses.
6. **Windows Application Control (WDAC) DLL Crash Mitigated & Consolidated:** Configured pure-Python fallback for `psycopg` in `app/__init__.py`, preventing fatal `ImportError` crashes when compiled C extensions are blocked by Windows system security policies.
7. **Test Suite Coverage Expanded:** Eliminated dormant/unused test imports (`Token`, `ScanRequest`, `UserCreate`) and added 17 new test cases. The test suite expanded from **11 tests to 28 tests** (100% passing), adding comprehensive coverage for user creation, timing attack mitigation, database CRUD, JWT authentication errors, database transaction rollbacks on failure, negation handling, internal whitespace validation, and Insecure Direct Object Reference (IDOR) isolation using an in-memory SQLite test database.

---

## 2. Original Findings and Severity

The findings were cataloged across six key functional areas:

| ID | Component / Location | Description | Severity | Status |
|---|---|---|---|---|
| **AUD-01** | `app/services/scan_services.py` (`detect_input_type`) | Multiline job posts starting with `http` or text ending with `.png` were misidentified as URL or IMAGE. | **High** | **FIXED** |
| **AUD-02** | `app/schemas/scan.py` (`ScanRequest`) | Whitespace-only strings (e.g., `"          "`) bypassed `min_length=10` validation and were saved as empty scans. | **Medium** | **FIXED** |
| **AUD-03** | `app/routers/scan.py` (`delete_scan_endpoint`) | Endpoint returned `None` without `response_class=Response`, producing illegal `Content-Type: application/json` on HTTP 204. | **Medium** | **FIXED** |
| **AUD-04** | `app/core/database.py`, `app/__init__.py` | `psycopg-binary` crashes on Windows under AppControl policies unless `PSYCOPG_IMPL=python` is set. | **Medium** | **FIXED** |
| **AUD-05** | `tests/test_backend.py` | `Token`, `ScanRequest`, and `UserCreate` imported but never tested. Zero tests for database services, rollbacks, or IDOR. | **Medium** | **FIXED** |
| **AUD-06** | `app/services/scan_services.py` | Upfront registration fee (30 pts) categorized as "Low Risk" (score <= 30). Inconsistent verdict boundaries. | **High** | **FIXED** |
| **AUD-07** | `app/services/scan_services.py` | URLs and images are scored as raw strings; no fetching or OCR occurs. | **High** | **DOCUMENTED** (Placeholder) |
| **AUD-08** | `app/main.py`, `app/routers/auth.py` | No rate limiting middleware on `/auth/login` or `/users/`; no CORS middleware configured. | **Medium** | **DOCUMENTED** (Security Risk) |

---

## 3. Confirmed Bugs Fixed and Corrections Applied

### 3.1 Risk Scoring and Verdict Policy Recalibration (AUD-06)
- **Problem:** Demanding an upfront `"registration fee"` awarded only 30 points. `get_verdict(score)` classified `score <= 30` as `"Low Risk"`. Consequently, an explicit demand for an upfront fee—the most definitive indicator of an employment scam—was classified as "Low Risk". Furthermore, all 5 individual scam patterns in isolation scored <= 30 and produced "Low Risk".
- **Fix:** 
  1. Recalibrated `"registration fee"` in `SCAM_PATTERNS` to **60 points** (critical fraud indicator).
  2. Implemented and documented a clear 3-tier verdict policy in `get_verdict`:
     - **Low Risk (`score <= 20`):** Benign job postings (0 points) or isolated mild urgency/communication signals (e.g., "urgent hiring" [10], "whatsapp only" [15], "no interview" [20]).
     - **Medium Risk (`21 <= score <= 50`):** Suspicious postings with bold claims (e.g., "guaranteed job" [25]) or combined low/moderate signals (e.g., "urgent hiring" + "whatsapp only" [25], "guaranteed job" + "no interview" [45]).
     - **High Risk (`score >= 51`):** Critical fraud indicators like advance fees ("registration fee" [60]) or compounding severe flags (e.g., "guaranteed job" + "no interview" + "urgent hiring" [55], or all combined [capped at 100]).
  3. Added regression tests verifying boundary scores (0, 10, 20, 21, 35, 50, 51, 60, 75, 100) and confirming that the registration fee example produces **"High Risk"**.

### 3.2 Input Type Classification Logic (`app/services/scan_services.py`)
- **Problem:** Python's `urlparse` treats anything after a valid authority as path or query. If a user submitted a multiline job description starting with `https://...`, `detect_input_type` returned `InputType.URL`. Similarly, `stripped_content.lower().endswith((".jpg", ...))` checked only the tail of the string, causing long job descriptions mentioning an attached image (e.g., `"See attached screenshot.png"`) to be classified as `InputType.IMAGE`.
- **Fix:** Refined `detect_input_type` to enforce that:
  1. A standalone URL must not contain internal unencoded whitespace or newlines (`not any(c.isspace() for c in stripped_content)`), and must possess a valid scheme (`http`/`https`) and non-empty `netloc`.
  2. A standalone image input must not contain newlines, must be at most two words (allowing single filenames like `"document.jpg"` or `"job ad.png"`), and must end with a recognized image extension.
  3. All other multi-line or multi-word texts default to `InputType.TEXT`.
- **Verification:** Verified across 19 edge-case inputs: standalone URLs, multiline URLs, image filenames with spaces, text mentioning images, malformed URLs (`https://`, `http://`, `https:///path`), and whitespace.

### 3.3 Whitespace-Only and Internally Padded Scan Submission (`app/schemas/scan.py`)
- **Problem:** `ScanRequest` defined `content: str = Field(min_length=10, max_length=50000)`. Input consisting of whitespace-only strings (e.g. `"          "`) or internally padded strings (e.g. `"a        b"`) bypassed minimum length validation. Naive `len(v.strip()) < 10` failed to reject `"a        b"` because `v.strip()` only removed leading/trailing whitespace, retaining the 8 internal spaces.
- **Fix:** Refined `@field_validator("content")` to calculate total non-whitespace characters using `sum(1 for c in v if not c.isspace())`. If the non-whitespace count is 0, it raises `"Content cannot be empty or whitespace only"`; if fewer than 10, it raises `"Content must contain at least 10 non-whitespace characters"`.
- **Verification:** Unit and regression tests verify rejection for `"Too short"`, whitespace-only `"               "`, `"          ab"`, and internally padded `"a        b"`.

### 3.4 HTTP 204 No Content Response Violation (`app/routers/scan.py`)
- **Problem:** `delete_scan_endpoint` used `@router.delete("/{scan_id}", status_code=status.HTTP_204_NO_CONTENT)` and returned `None`. Because FastAPI defaults to `JSONResponse`, the resulting HTTP response included a `Content-Type: application/json` header and a body of `b""`. Under RFC 7230/9110, an HTTP 204 response must not include representation metadata (Content-Type) or a body.
- **Fix:** Added `response_class=Response` to `@router.delete` and returned `Response(status_code=status.HTTP_204_NO_CONTENT)`. This guarantees an empty body with zero illegal content headers.

### 3.5 Windows Application Control DLL Crash & Consolidation (`app/__init__.py`)
- **Problem:** Running the application or migrations on Windows with AppControl / WDAC policies caused `import psycopg` to crash with:
  `ImportError: DLL load failed while importing _psycopg: An Application Control policy has blocked this file.`
- **Fix:** Configured `os.environ.setdefault("PSYCOPG_IMPL", "python")` canonically in `app/__init__.py`. Because Python executes `__init__.py` before importing any sub-module, this safely applies to all database models, routers, and scripts without requiring duplicate declarations.

### 3.6 False-Positive Negation Handling in Scam Heuristics (`app/services/scan_services.py`)
- **Problem:** Innocent employer disclaimers and fraud warnings (e.g., *"There is NO registration fee required to apply"*, *"ACME Corp will never ask for any registration fees"*, or *"This is not a guaranteed job"*) matched naive substring checks, immediately adding 60 points and falsely branding benign postings as "High Risk".
- **Fix:** Implemented clause-aware negation suppression (`_is_negated_match`) and regex word-boundary matching (`_is_pattern_present` using `\b{pattern}s?\b`). The detector checks up to 4 words before the keyword for negation indicators (`no`, `never`, `without`, `not`, `zero`) within the same sentence/clause. If all matches of a pattern are negated, the score contribution is suppressed; unnegated occurrences or mixed postings still trigger the score.

---

## 4. Dead Code Removed & Safety Verification

1. **Unused Imports in `tests/test_backend.py` (`Token`, `ScanRequest`, `UserCreate`):**
   - *Investigation:* AST parsing revealed these three schema models were imported on lines 9–11 but never used in any assertions.
   - *Action:* Leveraged these models to build thorough unit tests (`test_token_schema`, `test_user_create_validation`, `test_scan_request_validation`).
2. **`InputType.IMAGE` Enum Value:**
   - *Investigation:* Analyzed whether `InputType.IMAGE` was dead code. It is referenced in `app/models/scan.py`, `app/schemas/scan.py`, and `app/services/scan_services.py`, and verified in existing unit tests. Removing it would break existing API contracts and schema validation. It was safely retained as a planned placeholder.
3. **Alembic Model Registrations in `alembic/env.py`:**
   - *Investigation:* `from app.models.user import User` and `from app.models.scan import Scan` appeared unused by AST analysis. However, importing them is necessary to attach table metadata to `Base.metadata` for Alembic autogeneration. Both imports were preserved.

---

## 5. Files Changed and Justification

| File Path | Nature of Change | Justification |
|---|---|---|
| [`app/__init__.py`](file:///C:/Users/vasista0305/OneDrive/Desktop/AI-JOB-SCAM-DETECTOR/app/__init__.py) | Added `os.environ.setdefault("PSYCOPG_IMPL", "python")` | Ensures the pure-Python driver fallback is active upon any import of the `app` package. Consolidated here so downstream sub-modules do not need duplicate configurations. |
| [`app/schemas/scan.py`](file:///C:/Users/vasista0305/OneDrive/Desktop/AI-JOB-SCAM-DETECTOR/app/schemas/scan.py) | Added `field_validator("content")` to `ScanRequest` | Rejects empty, whitespace-only, and internally padded submissions (e.g. `"a        b"`); enforces minimum 10 non-whitespace characters across the entire content string. |
| [`app/services/scan_services.py`](file:///C:/Users/vasista0305/OneDrive/Desktop/AI-JOB-SCAM-DETECTOR/app/services/scan_services.py) | 1. Hardened `detect_input_type`<br>2. Recalibrated `"registration fee"` to 60<br>3. Added clause-aware negation & word-boundary matching<br>4. Documented and updated `get_verdict` policy | 1. Eliminates false-positive URL and IMAGE classifications.<br>2. Corrects the critical scoring flaw where upfront fees were marked "Low Risk".<br>3. Prevents legitimate disclaimers ("no registration fee required") from triggering false positives.<br>4. Implements clear Low (0-20), Medium (21-50), and High (51-100) boundaries. |
| [`app/routers/scan.py`](file:///C:/Users/vasista0305/OneDrive/Desktop/AI-JOB-SCAM-DETECTOR/app/routers/scan.py) | Added `response_class=Response` and returned `Response(status_code=204)` | Conforms to RFC 7230/9110 by omitting Content-Type and payload on 204 No Content. |
| [`tests/test_backend.py`](file:///C:/Users/vasista0305/OneDrive/Desktop/AI-JOB-SCAM-DETECTOR/tests/test_backend.py) | Added 17 new test cases (28 total) across schemas, services, IDOR, boundaries, negation, and rollbacks | Validates all fixes, boundary conditions, registration fee scoring, transaction rollbacks, negation handling, internal whitespace rejection, and security boundaries. |

---

## 6. Test Commands and Actual Results

### 6.1 Execution Command
```powershell
.\venv\Scripts\python.exe -m unittest -v tests/test_backend.py
```

### 6.2 Pre-Fix Test Baseline
- **Tests Discovered:** 11
- **Passed:** 11
- **Failed:** 0
- **Duration:** 3.146s
- **Coverage:** Only isolated unit tests for hashing, JWT format, and heuristic scores. Zero tests for database operations, CRUD logic, user authentication, or IDOR protection. Required `$env:PSYCOPG_IMPL="python"` to avoid immediate crash.

### 6.3 Post-Fix Test Results
- **Tests Discovered:** 28
- **Passed:** 28
- **Failed:** 0
- **Errors:** 0
- **Duration:** 13.322s
- **Environment:** Executed cleanly without manually setting any environment variables.

#### Detailed Test Inventory:
```text
test_routes_registered (tests.test_backend.TestAppRoutes.test_routes_registered) ... ok
test_jwt_token_creation_and_payload (tests.test_backend.TestAuth.test_jwt_token_creation_and_payload) ... ok
test_database_error_rollbacks (tests.test_backend.TestDatabaseServices.test_database_error_rollbacks) ... ok
test_scan_services_and_idor (tests.test_backend.TestDatabaseServices.test_scan_services_and_idor) ... ok
test_user_services_flow (tests.test_backend.TestDatabaseServices.test_user_services_flow) ... ok
test_delete_scan_endpoint_not_found (tests.test_backend.TestDeleteScanEndpoint.test_delete_scan_endpoint_not_found) ... ok
test_delete_scan_endpoint_success (tests.test_backend.TestDeleteScanEndpoint.test_delete_scan_endpoint_success) ... ok
test_get_current_user_expired_token (tests.test_backend.TestDependencies.test_get_current_user_expired_token) ... ok
test_get_current_user_invalid_sub (tests.test_backend.TestDependencies.test_get_current_user_invalid_sub) ... ok
test_get_current_user_invalid_token (tests.test_backend.TestDependencies.test_get_current_user_invalid_token) ... ok
test_get_current_user_missing_sub (tests.test_backend.TestDependencies.test_get_current_user_missing_sub) ... ok
test_get_current_user_nonexistent_user (tests.test_backend.TestDependencies.test_get_current_user_nonexistent_user) ... ok
test_get_current_user_valid_token (tests.test_backend.TestDependencies.test_get_current_user_valid_token) ... ok
test_calculate_risk_score (tests.test_backend.TestScanLogic.test_calculate_risk_score) ... ok
test_calculate_risk_score_capped_at_100 (tests.test_backend.TestScanLogic.test_calculate_risk_score_capped_at_100) ... ok
test_calculate_risk_score_negation_handling (tests.test_backend.TestScanLogic.test_calculate_risk_score_negation_handling) ... ok
test_detect_input_type (tests.test_backend.TestScanLogic.test_detect_input_type) ... ok
test_detect_input_type_edge_cases (tests.test_backend.TestScanLogic.test_detect_input_type_edge_cases) ... ok
test_registration_fee_yields_high_risk (tests.test_backend.TestScanLogic.test_registration_fee_yields_high_risk) ... ok
test_verdicts (tests.test_backend.TestScanLogic.test_verdicts) ... ok
test_scan_request_validation (tests.test_backend.TestSchemas.test_scan_request_validation) ... ok
test_scan_response_from_orm (tests.test_backend.TestSchemas.test_scan_response_from_orm) ... ok
test_token_schema (tests.test_backend.TestSchemas.test_token_schema) ... ok
test_user_create_validation (tests.test_backend.TestSchemas.test_user_create_validation) ... ok
test_user_response_from_orm (tests.test_backend.TestSchemas.test_user_response_from_orm) ... ok
test_dummy_hash_for_timing_attacks (tests.test_backend.TestSecurity.test_dummy_hash_for_timing_attacks) ... ok
test_normalize_email (tests.test_backend.TestSecurity.test_normalize_email) ... ok
test_password_hashing_and_verification (tests.test_backend.TestSecurity.test_password_hashing_and_verification) ... ok

----------------------------------------------------------------------
Ran 28 tests in 13.322s

OK
```

---

## 7. Database and Alembic Consistency Findings

### 7.1 Migration Revision History
Executed `alembic history`:
- `6485f0c23715` (Initial users table)
- `a97c23a4b382` (Initial scans table)
- `b1a5e08f361c` (Head: Sync scan schema, add cascade delete, add indexes)

The revision chain is strictly linear, with unambiguous down-revisions and zero branched heads.

### 7.2 Offline SQL Verification
Executed:
```powershell
.\venv\Scripts\python.exe -m alembic upgrade base:head --sql
```
- **Result:** Successfully rendered full PostgreSQL DDL.
- **Foreign Key Constraints:** `scans.user_id` enforces `ON DELETE CASCADE` referencing `users.id`.
- **Indexes:** `ix_users_email` (unique index on `users.email`) and `ix_scans_user_id` (index on `scans.user_id`) are generated as specified.
- **Model-to-Schema Agreement:** Declarative models in `app/models/user.py` and `app/models/scan.py` are in complete synchronization with migration head `b1a5e08f361c`. No schema changes or new migrations were necessary.

---

## 8. Remaining Architectural Limitations & Blockers

The following items are confirmed architectural limitations that require future phases and are intentionally out-of-scope for codebase stabilization:

1. **Advertised Multi-Modal Capabilities Are Non-Functional (AUD-07):**
   - Submitting URLs or image filenames merely executes substring searches against the URL or filename string. No HTTP scraping, SSRF defenses, or OCR extraction exist.
   - *Requirement:* Implement in Phase 3 (Safe URL Analysis) and Phase 4 (Image Upload & OCR).
2. **Absence of Rate Limiting (AUD-08):**
   - No throttling middleware exists on `/auth/login` or `/users/`.
   - *Requirement:* Introduce `slowapi` with in-memory or Redis-backed rate limiting.
3. **Missing CORS Configuration (AUD-08):**
   - No `CORSMiddleware` is registered in `app/main.py`. External web frontends or extensions will be blocked by browser CORS policies until configured.

---

## 9. Issues Requiring Manual Verification on Host Machine / PostgreSQL

The following verifications could not be executed locally due to environment constraints and require manual verification against a live PostgreSQL service:

1. **Live PostgreSQL Network Connectivity & Migration Execution:**
   - Testing utilized SQLite in-memory engines (`sqlite:///:memory:`). The local PostgreSQL service on `localhost:5432` was inactive in this environment.
   - *Action Required:* Run `alembic upgrade head` against a live PostgreSQL 15+ database to verify table and index creation live.
2. **Bcrypt Work-Factor Latency Under High Concurrency:**
   - Timing attack mitigation via `DUMMY_HASH` was verified functionally in unit tests. Load-testing with concurrent unauthenticated requests over live network sockets should be measured to confirm CPU headroom.
3. **Integration Verification via HTTP Client:**
   - Endpoints were tested via unit and service harnesses. End-to-end HTTP integration tests requiring `httpx` / `pytest` should be executed once development dependencies are installed.

---

## 10. Final Readiness Assessment

| Category | Assessment | Status |
|---|---|---|
| **Codebase Stability** | Core backend logic, routing, schemas, and session handling are stabilized. | **STABILIZED** |
| **Heuristic Scoring & Verdicts** | Flawed threshold resolved; "registration fee" produces High Risk; boundaries tested. | **CORRECTED** |
| **Data Integrity & Isolation** | User-scoped scan queries, rollbacks, and cascade deletions verified with automated IDOR tests. | **PASSED** |
| **API Contract Integrity** | Existing endpoints preserve expected inputs, outputs, and status codes. HTTP 204 compliant. | **PASSED** |
| **Test Suite Health** | 28 automated tests pass cleanly with 0 failures, 0 errors, and 0 warnings. | **PASSED** |
| **Readiness for Feature Development** | Safe and solid foundation for future URL scraping, OCR, and ML/LLM pipelines. | **READY** |