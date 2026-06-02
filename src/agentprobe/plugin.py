import pytest
from ._session import Session


@pytest.fixture
def agentprobe():
    """pytest fixture — provides a Session for record/replay agent testing.

    Usage::

        def test_my_agent(agentprobe):
            with agentprobe.replay("tests/fixtures/my_session.jsonl") as probe:
                result = my_agent.run("list files in /tmp")
                probe.assert_tool_called("bash")
                probe.assert_max_iterations(3)
    """
    return Session()
