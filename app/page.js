'use client';
/**
 * MAIN PAGE (React Component)
 * ===========================
 * 'use client' at the top tells Next.js: "this component runs in the BROWSER".
 *
 * WHY DOES THAT MATTER?
 * Next.js by default renders components on the SERVER (fast initial load, good
 * for SEO). But interactive things — file uploads, clicking buttons, typing
 * in a chat box — need to run in the browser because they respond to user actions.
 * 'use client' switches this component to browser mode.
 *
 * WHAT IS A REACT COMPONENT?
 * A function that returns HTML (written as JSX). React re-renders it whenever
 * its STATE changes.
 *
 * WHAT IS STATE?
 * State is data that, when it changes, causes the component to re-render (update
 * the UI automatically). We use useState() to create state variables.
 *
 * Example: when the user types a message, we update the 'input' state variable.
 * React automatically re-renders the input field to show what they typed.
 */

import { useState, useRef, useEffect } from 'react';

export default function ChatPage() {
  // STATE VARIABLES
  // Each useState() returns [currentValue, functionToUpdateIt]
  const [document, setDocument] = useState(null);      // The uploaded document info
  const [messages, setMessages] = useState([]);         // Chat history
  const [input, setInput] = useState('');               // Current message being typed
  const [isUploading, setIsUploading] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [showSources, setShowSources] = useState(null); // Which message's sources to show

  const messagesEndRef = useRef(null);                  // For auto-scrolling to bottom
  const fileInputRef = useRef(null);                    // Hidden file input element

  // Auto-scroll to the latest message whenever messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  /**
   * HANDLE FILE UPLOAD
   * When user selects a file, this function:
   * 1. Sends the file to POST /api/upload
   * 2. Waits for the server to process it (extract text, create embeddings)
   * 3. Saves the document info in state so we can use it for questions
   */
  async function handleUpload(file) {
    if (!file) return;

    setIsUploading(true);
    setDocument(null);
    setMessages([]);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
        // Note: don't set Content-Type header manually for FormData!
        // The browser sets it automatically with the correct "boundary" parameter
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Upload failed');
      }

      setDocument(data);
      setMessages([{
        role: 'assistant',
        content: `I've read **${data.filename}** and created ${data.chunkCount} searchable chunks. Ask me anything about it!`,
        isIntro: true,
      }]);

    } catch (err) {
      alert(`Upload failed: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  }

  /**
   * HANDLE DRAG AND DROP
   * Users can drag a file onto the upload area instead of clicking.
   */
  function handleDrop(e) {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  }

  /**
   * HANDLE SENDING A QUESTION
   * Sends the question + documentId to POST /api/ask
   * The server runs the RAG pipeline and returns an answer.
   */
  async function handleSend() {
    if (!input.trim() || !document || isAsking) return;

    const question = input.trim();
    setInput('');

    // Immediately show the user's message in the chat
    setMessages(prev => [...prev, { role: 'user', content: question }]);
    setIsAsking(true);

    try {
      const response = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          documentId: document.documentId,
          question,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to get answer');
      }

      // Add AI response to chat, including cache metadata
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        sourceChunks: data.sourceChunks,
        fromCache: data.fromCache,
        cacheSource: data.cacheSource,
        cacheSimilarity: data.cacheSimilarity,
        matchedQuestion: data.matchedQuestion,
        tokensUsed: data.tokensUsed,
      }]);

    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${err.message}`,
        isError: true,
      }]);
    } finally {
      setIsAsking(false);
    }
  }

  // Send message on Enter key (but Shift+Enter = new line)
  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto">

      {/* HEADER */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-gray-800">
        <div>
          <h1 className="text-xl font-bold text-white">Chat with Your Document</h1>
          <p className="text-xs text-gray-500 mt-0.5">RAG-powered — answers come only from your document</p>
        </div>
        {document && (
          <div className="text-right">
            <p className="text-sm font-medium text-indigo-400">{document.filename}</p>
            <p className="text-xs text-gray-500">{document.chunkCount} chunks indexed</p>
          </div>
        )}
      </header>

      {/* MAIN AREA */}
      <div className="flex flex-1 overflow-hidden">

        {/* UPLOAD PANEL — shown before a document is uploaded */}
        {!document && (
          <div className="flex flex-1 items-center justify-center p-8">
            <div
              className="w-full max-w-md border-2 border-dashed border-gray-700 rounded-2xl p-10 text-center cursor-pointer hover:border-indigo-500 hover:bg-gray-900 transition-all"
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.docx,.txt"
                className="hidden"
                onChange={(e) => handleUpload(e.target.files[0])}
              />

              {isUploading ? (
                <div>
                  <div className="w-12 h-12 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
                  <p className="text-gray-300 font-medium">Processing document...</p>
                  <p className="text-gray-500 text-sm mt-1">Extracting text and creating embeddings</p>
                </div>
              ) : (
                <div>
                  <div className="text-5xl mb-4">📄</div>
                  <p className="text-white font-semibold text-lg mb-1">Drop your document here</p>
                  <p className="text-gray-400 text-sm mb-4">or click to browse</p>
                  <p className="text-gray-600 text-xs">Supports PDF, DOCX, and TXT</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* CHAT PANEL — shown after a document is uploaded */}
        {document && (
          <div className="flex flex-col flex-1 overflow-hidden">

            {/* Messages list */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              {messages.map((msg, i) => (
                <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[80%] ${msg.role === 'user' ? 'order-2' : 'order-1'}`}>

                    {/* Message bubble */}
                    <div className={`rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
                      msg.role === 'user'
                        ? 'bg-indigo-600 text-white'
                        : msg.isError
                        ? 'bg-red-900/40 text-red-300 border border-red-800'
                        : 'bg-gray-800 text-gray-100'
                    }`}>
                      {/* Render **bold** markdown manually */}
                      {msg.content.split(/(\*\*[^*]+\*\*)/).map((part, j) =>
                        part.startsWith('**') && part.endsWith('**')
                          ? <strong key={j}>{part.slice(2, -2)}</strong>
                          : part
                      )}
                    </div>

                    {/* Cache / token badge */}
                    {msg.role === 'assistant' && !msg.isIntro && !msg.isError && (
                      <div className="mt-1 flex items-center gap-2">
                        {msg.fromCache ? (
                          <span className="text-[10px] bg-green-900/40 text-green-400 border border-green-800 px-2 py-0.5 rounded-full">
                            ⚡ {msg.cacheSource === 'predefined' ? 'Predefined' : 'Cached'} · {Math.round((msg.cacheSimilarity||0)*100)}% match · 0 tokens
                          </span>
                        ) : (
                          <span className="text-[10px] bg-gray-800 text-gray-500 border border-gray-700 px-2 py-0.5 rounded-full">
                            🤖 LLM · {msg.tokensUsed || '?'} tokens used · auto-cached
                          </span>
                        )}
                      </div>
                    )}

                    {/* "Show sources" button for AI messages */}
                    {msg.sourceChunks?.length > 0 && (
                      <div className="mt-2">
                        <button
                          onClick={() => setShowSources(showSources === i ? null : i)}
                          className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors"
                        >
                          {showSources === i ? '▲ Hide' : '▼ Show'} {msg.sourceChunks.length} source excerpt{msg.sourceChunks.length !== 1 ? 's' : ''}
                        </button>

                        {/* Source chunks panel */}
                        {showSources === i && (
                          <div className="mt-2 space-y-2">
                            {msg.sourceChunks.map((chunk, j) => (
                              <div key={j} className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-xs text-gray-400 leading-relaxed">
                                <span className="text-indigo-500 font-mono text-[10px] block mb-1">EXCERPT {j + 1}</span>
                                {chunk}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                  </div>
                </div>
              ))}

              {/* Typing indicator */}
              {isAsking && (
                <div className="flex justify-start">
                  <div className="bg-gray-800 rounded-2xl px-4 py-3">
                    <div className="flex space-x-1">
                      {[0, 1, 2].map(i => (
                        <div key={i} className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />
                      ))}
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* INPUT BAR */}
            <div className="border-t border-gray-800 p-4">
              <div className="flex gap-3 items-end">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask anything about your document..."
                  rows={1}
                  className="flex-1 bg-gray-800 text-gray-100 placeholder-gray-500 rounded-xl px-4 py-3 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500 border border-gray-700"
                  style={{ minHeight: '44px', maxHeight: '120px' }}
                  onInput={(e) => {
                    e.target.style.height = 'auto';
                    e.target.style.height = e.target.scrollHeight + 'px';
                  }}
                />
                <button
                  onClick={handleSend}
                  disabled={!input.trim() || isAsking}
                  className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 disabled:text-gray-500 text-white px-4 py-3 rounded-xl text-sm font-medium transition-colors flex-shrink-0"
                >
                  Send
                </button>
              </div>
              <p className="text-xs text-gray-600 mt-2 text-center">
                Enter to send · Shift+Enter for new line
              </p>

              {/* Upload different document link */}
              <div className="text-center mt-2">
                <button
                  onClick={() => { setDocument(null); setMessages([]); }}
                  className="text-xs text-gray-600 hover:text-gray-400 transition-colors"
                >
                  Upload a different document
                </button>
              </div>
            </div>

          </div>
        )}

      </div>
    </div>
  );
}
