from unittest.mock import patch, MagicMock
from app.services.llm.agent import get_agent

@patch('app.services.llm.agent.get_system_prompt')
@patch('app.services.llm.agent.ChatOpenAI')
@patch('app.services.llm.agent.create_agent')
def test_get_agent(mock_create_agent, mock_chat_openai, mock_get_prompt):
    """Test agent creation"""
    mock_llm = MagicMock()
    mock_chat_openai.return_value = mock_llm

    mock_agent = MagicMock()
    mock_create_agent.return_value = mock_agent

    mock_get_prompt.return_value = [{"role": "system", "content": "Test prompt"}]

    agent = get_agent()

    assert agent is not None
    mock_create_agent.assert_called_once()
