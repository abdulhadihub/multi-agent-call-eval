from ui.audit import group_audit_events


def test_audit_events_group_by_tool_call():
    events = [
        {"tool_call_id": "c1", "event_type": "proposed", "tool": "create_ticket", "agent": "action", "created_at": "1", "detail": {}},
        {"tool_call_id": "c1", "event_type": "approved", "tool": "create_ticket", "agent": "approval", "created_at": "2", "detail": {}},
        {"tool_call_id": "c1", "event_type": "executed", "tool": "create_ticket", "agent": "executor", "created_at": "3", "detail": {}},
        {"tool_call_id": "c2", "event_type": "denied", "tool": "issue_refund", "agent": "executor", "created_at": "4", "detail": {}},
    ]
    groups = group_audit_events(events)
    assert len(groups) == 2
    assert [e["event_type"] for e in groups[0]["events"]] == ["proposed", "approved", "executed"]
    assert groups[1]["status"] == "denied"
