import os

import pytest

from ._session import MultiSession, Session


def pytest_addoption(parser):
    parser.addoption(
        "--agentprobe-update",
        action="store_true",
        default=False,
        help="Force re-record all agentprobe fixtures (equivalent to AGENTPROBE_UPDATE=1)",
    )


def pytest_configure(config):
    try:
        if config.getoption("--agentprobe-update"):
            os.environ["AGENTPROBE_UPDATE"] = "1"
    except ValueError:
        pass  # option not registered (e.g. when called from programmatic config)


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


@pytest.fixture
def agentprobe_multi():
    """pytest fixture — provides a MultiSession for multi-client agent testing.

    Usage::

        def test_orchestrator(agentprobe_multi):
            client_main = openai.OpenAI(api_key="...")
            client_sub  = openai.OpenAI(api_key="...")
            with agentprobe_multi.replay(client_main, "fixtures/main.jsonl") as pm:
                with agentprobe_multi.replay(client_sub, "fixtures/sub.jsonl") as ps:
                    run_orchestrator(client_main, client_sub)
            pm.assert_tool_called("bash")
    """
    return MultiSession()
