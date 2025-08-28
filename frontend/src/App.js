import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid'; // For generating unique session IDs
import './App.css';

function App() {
    const [currentChatSessionId, setCurrentChatSessionId] = useState(null);
    const [chatSessions, setChatSessions] = useState([]);
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    // Use useCallback to memoize fetchChatSessions
    const fetchChatSessions = useCallback(async () => {
        try {
            const { data } = await axios.get('http://localhost:8000/chatsessions');
            setChatSessions(data);
        } catch (error) {
            console.error('Error fetching chat sessions:', error);
        }
    }, []); // Empty dependency array means this function is created once

    const fetchMessages = useCallback(async (sessionId) => {
        setIsLoading(true);
        try {
            const response = await axios.get(`http://localhost:8000/chatsessions/${sessionId}/messages`);
            // Parse response_data for bot messages and ensure text_content is mapped to 'text' for display
            const formattedMessages = response.data.map(msg => ({
                ...msg,
                text: msg.text_content, // Explicitly map text_content to text for display
                response: msg.sender === 'bot' && msg.response_data ? JSON.parse(msg.response_data) : null,
            }));
            setMessages(formattedMessages);
        } catch (error) {
            console.error('Error fetching messages:', error);
            setMessages([]);
        } finally {
            setIsLoading(false);
        }
    }, []);

    const startNewChatSession = useCallback(async () => {
        try {
            const newSessionId = uuidv4();
            const { data: newSession } = await axios.post('http://localhost:8000/chatsessions', {
                session_id_uuid: newSessionId,
                title: "New Chat"
            });
            setCurrentChatSessionId(newSession.session_id_uuid);
            setMessages([]); // Clear messages for the new chat
            fetchChatSessions(); // Refresh the sidebar with the new session
        } catch (error) {
            console.error('Error starting new chat session:', error);
        }
    }, [fetchChatSessions]); // fetchChatSessions is a dependency as it's called inside

    const handleSessionClick = (session_id_uuid) => {
        setCurrentChatSessionId(session_id_uuid);
        // fetchMessages will be called by the useEffect when currentChatSessionId changes
    };

    useEffect(() => {
        if (currentChatSessionId) {
            fetchMessages(currentChatSessionId);
        } else if (chatSessions.length > 0) {
            // If there are existing sessions, select the latest one
            setCurrentChatSessionId(chatSessions[0].session_id_uuid);
        } else {
            // If no sessions exist, create a new one on initial load
            startNewChatSession();
        }
    }, [currentChatSessionId, chatSessions, startNewChatSession, fetchMessages]);

    useEffect(() => {
        fetchChatSessions();
    }, [fetchChatSessions]);

    const sendMessage = async () => {
        if (input.trim() === '') return;
        if (!currentChatSessionId) {
            console.error('No active chat session. Please start a new chat.');
            return;
        }

        const userQuery = input;
        let extractedPrice = null;
        let extractedDate = null;

        // Regex to find price (e.g., "price 12345" or "Rs 12345")
        const priceMatch = userQuery.match(/(?:price|rs|Rs|INR)\s*(\d+)/i);
        if (priceMatch) {
            extractedPrice = priceMatch[1];
        }

        // Regex to find date (e.g., "date 14-08-2025" or "on 14-08-2025")
        const dateMatch = userQuery.match(/(?:date|on|by)\s*(\d{2}-\d{2}-\d{4})/i);
        if (dateMatch) {
            extractedDate = dateMatch[1];
        }

        // Optimistically add user message to UI
        const userMessageForDisplay = { sender: 'user', text: userQuery, price: extractedPrice, date: extractedDate };
        setMessages((prevMessages) => [...prevMessages, userMessageForDisplay]);
        setIsLoading(true);

        try {
            const { data: botResponseData } = await axios.post('http://localhost:8000/chat', {
                user_query: userQuery,
                session_id_uuid: currentChatSessionId, // Send current session ID to backend
                price: extractedPrice,
                date: extractedDate,
            });

            const botMessageForDisplay = {
                sender: 'bot',
                text: botResponseData.relevant_sections[0]?.display_text || "No relevant sections found.", // Display summary
                response: botResponseData, // Store full response for detailed display
            };

            setMessages((prevMessages) => [...prevMessages, botMessageForDisplay]);
            // After successful message, refresh chat sessions to update titles/timestamps
            fetchChatSessions(); 

        } catch (error) {
            console.error('Error sending message:', error);
            setMessages((prevMessages) => [
                ...prevMessages,
                { sender: 'bot', text: 'Error: Could not connect to the backend or process your request.' },
            ]);
        } finally {
            setInput('');
            setIsLoading(false);
        }
    };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Legal QA Chatbot</h1>
      </header>
      <div className="chat-container">
        {/* Chat history sidebar */}
        <div className="chat-sidebar">
          <button onClick={startNewChatSession} className="new-chat-button">+ New chat</button>
          <div className="chat-history-list">
            {chatSessions.map((session) => (
              <div
                key={session.session_id_uuid} // Use UUID as key
                className={`chat-history-item ${session.session_id_uuid === currentChatSessionId ? 'active' : ''}`}
                onClick={() => handleSessionClick(session.session_id_uuid)}
              >
                {session.title}
              </div>
            ))}
          </div>
        </div>
        <div className="main-content">
          <div className="messages-display">
            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.sender}`}>
                {msg.sender === 'user' ? (
                  <>
                    <p><strong>You:</strong> {msg.text}</p>
                    {msg.price && <p><strong>Price:</strong> {msg.price}</p>}
                    {msg.date && <p><strong>Date:</strong> {msg.date}</p>}
                  </>
                ) : (
                  <div className="bot-response">
                    <p><strong>Bot:</strong></p>
                    {msg.response && msg.response.predicted_intent && (
                      <p><strong>Intent:</strong> {msg.response.predicted_intent}</p>
                    )}
                    {msg.response && msg.response.relevant_sections && (
                      <div>
                        <h3>Relevant Sections:</h3>
                        {/* Map through relevant_sections from bot's full response */}
                        {msg.response.relevant_sections.map((section, secIndex) => (
                          <div key={secIndex} className="section-card">
                            <h4>{section.section_id} - {section.chapter}</h4>
                            <p><strong>Summary:</strong> {section.display_text}</p>
                            {section.examples && section.examples.length > 0 && (
                              <p><strong>Examples:</strong> {section.examples.join(', ')}</p>
                            )}
                            <p><em>(Distance: {section.distance.toFixed(4)})</em></p>
                            <p><em>Original Text Excerpt:</em> {section.original_text_excerpt}</p>
                          </div>
                        ))}
                      </div>
                    )}
                    {msg.response && msg.response.rule_engine_analysis && (
                      <div className="rule-engine-analysis">
                        <h3>Rule Engine Analysis:</h3>
                        <p><strong>Recommended Forum:</strong> {msg.response.rule_engine_analysis.recommended_forum}</p>
                        <p><strong>Eligibility Status:</strong> {msg.response.rule_engine_analysis.eligibility_status}</p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
            {isLoading && <div className="message bot"><strong>Bot:</strong> Thinking...</div>}
          </div>
          <div className="input-area">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && !isLoading && sendMessage()}
              placeholder="Type your legal query (e.g., 'My phone is faulty, price 25000, date 01-01-2023')..."
              disabled={isLoading}
            />
            <button onClick={sendMessage} disabled={isLoading}>
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
