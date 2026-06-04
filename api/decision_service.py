from datetime import datetime, timezone

from api.schemas import DecisionInput

DECISION_DISPLAY = {
    "Approved": {
        "title": "Pengajuan diterima",
        "description": "Rekomendasi approval telah dicatat untuk proses lanjutan.",
    },
    "Revision Requested": {
        "title": "Revisi plafon sedang diajukan",
        "description": "Pengajuan menunggu tindak lanjut atas plafon yang direvisi.",
    },
    "Rejected": {
        "title": "Pengajuan ditolak",
        "description": "Rekomendasi penolakan telah dicatat oleh analis.",
    },
}

decision_store = {}


def record_decision(data: DecisionInput):
    display = DECISION_DISPLAY[data.status]
    record = {
        "merchant_id": data.merchant_id,
        "status": data.status,
        "display_status": display["title"],
        "description": display["description"],
        "note": data.note,
        "revision_limit": data.revision_limit,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    decision_store[data.merchant_id] = record
    return record
