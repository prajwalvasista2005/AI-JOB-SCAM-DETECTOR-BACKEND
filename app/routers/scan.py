from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user, get_db
from app.core.rate_limit import scan_rate_limiter
from app.models.user import User
from app.schemas.scan import AnalysisMode, ScanRequest, ScanResponse, Verdict
from app.services.scan_services import (
    ScanValidationError,
    create_scan,
    delete_scan,
    get_scan_by_id,
    get_user_scans,
)

router = APIRouter(prefix="/scan", tags=["Scan"])


@router.post("/", response_model=ScanResponse, status_code=status.HTTP_201_CREATED)
def create_scan_endpoint(
    scan_request: ScanRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(scan_rate_limiter),
):
    try:
        return create_scan(content=scan_request.content, user_id=current_user.id, db=db)
    except ScanValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


@router.get("/", response_model=list[ScanResponse], status_code=status.HTTP_200_OK)
def get_user_scans_endpoint(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    analysis_mode: AnalysisMode | None = Query(default=None),
    verdict: Verdict | None = Query(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    actual_limit = limit.default if hasattr(limit, "default") else limit
    actual_offset = offset.default if hasattr(offset, "default") else offset
    actual_mode = analysis_mode.default if hasattr(analysis_mode, "default") else analysis_mode
    actual_verdict = verdict.default if hasattr(verdict, "default") else verdict

    return get_user_scans(
        user_id=current_user.id,
        db=db,
        limit=actual_limit,
        offset=actual_offset,
        analysis_mode=actual_mode,
        verdict=actual_verdict,
    )


@router.get("/{scan_id}", response_model=ScanResponse, status_code=status.HTTP_200_OK)
def get_scan_by_id_endpoint(
    scan_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    scan = get_scan_by_id(scan_id=scan_id, user_id=current_user.id, db=db)
    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scan not found",
        )
    return scan


@router.delete("/{scan_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_scan_endpoint(
    scan_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    success = delete_scan(scan_id=scan_id, user_id=current_user.id, db=db)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scan not found",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
