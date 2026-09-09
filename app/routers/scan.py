from fastapi import APIRouter
from app.schemas.scan import ScanRequest, ScanResponse

router = APIRouter(tags=["Scan"])


@router.post("/scan", response_model=ScanResponse)
def scan(request: ScanRequest):
    return {
        "input_type": request.input_type,
        "content": request.content
    }
