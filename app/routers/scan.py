from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user, get_db
from app.models.user import User
from app.schemas.scan import ScanRequest, ScanResponse
from app.services.scan_services import (
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
    current_user: User = Depends(get_current_user),
):
    return create_scan(content=scan_request.content, user_id=current_user.id, db=db)


@router.get("/", response_model=list[ScanResponse], status_code=status.HTTP_200_OK)
def get_user_scans_endpoint(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_user_scans(user_id=current_user.id, db=db)


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


@router.delete("/{scan_id}", status_code=status.HTTP_204_NO_CONTENT)
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
    return None
