import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [price, setPrice] = useState('');
  const [date, setDate] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async () => {
    if (input.trim() === '') return;

    const newMessages = [...messages, { sender: 'user', text: input, price, date }];
    setMessages(newMessages);
    setIsLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/chat', {
        user_query: input,
        price: price || null, // Send null if empty
        date: date || null,   // Send null if empty
      });

      const botMessage = {
        sender: 'bot',
        response: response.data,
      };
      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: 'bot', text: 'Error: Could not connect to the backend or process your request.' },
      ]);
    } finally {
      setInput('');
      setPrice('');
      setDate('');
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Legal QA Chatbot</h1>
      </header>
      <div className="chat-container">
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
            placeholder="Type your legal query..."
            disabled={isLoading}
          />
          <input
            type="text"
            value={price}
            onChange={(e) => setPrice(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !isLoading && sendMessage()}
            placeholder="Price (e.g., 34000)"
            disabled={isLoading}
          />
          <input
            type="text"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !isLoading && sendMessage()}
            placeholder="Date (DD-MM-YYYY, e.g., 14-08-2025)"
            disabled={isLoading}
          />
          <button onClick={sendMessage} disabled={isLoading}>
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
