import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Activity, ChevronUp, ChevronDown } from 'lucide-react';
import Sidebar from '../components/Sidebar';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const ChatPage = ({
    apiBase,
    user,
    usage,
    uploading,
    file,
    handleFileUpload,
    documents,
    docId,
    setDocId,
    setFile,
    messages,
    setMessages,
    loading,
    query,
    setQuery,
    handleQuery,
    chatEndRef
}) => {
    const [showModels, setShowModels] = React.useState(false);



    return (
        <>
            <Sidebar
                user={user}
                usage={usage}
                uploading={uploading}
                file={file}
                handleFileUpload={handleFileUpload}
            />

            {/* Chat Area */}
            <div className="glass chat-section">
                <div className="messages-container">
                    <AnimatePresence>
                        {messages.map((msg, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                className={`message ${msg.role}`}
                            >
                                <ReactMarkdown
                                    remarkPlugins={[remarkGfm]}
                                >
                                    {msg.content}
                                </ReactMarkdown>
                                
                                {msg.confidence_score !== undefined && (
                                    <div style={{ marginTop: '10px', padding: '8px', background: 'rgba(0,0,0,0.1)', borderRadius: '6px', fontSize: '12px', borderLeft: '2px solid var(--groq-orange)' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                            <span style={{ fontWeight: 'bold', color: 'var(--groq-orange)' }}>
                                                STCA Confidence: {msg.confidence_score}%
                                            </span>
                                            {msg.source_format && (
                                                <span style={{ 
                                                    fontSize: '10px', 
                                                    padding: '2px 6px', 
                                                    background: 'rgba(255,255,255,0.1)', 
                                                    borderRadius: '4px',
                                                    textTransform: 'uppercase',
                                                    opacity: 0.8
                                                }}>
                                                    SOURCE: {msg.source_format}
                                                </span>
                                            )}
                                        </div>
                                        {msg.reasoning && (
                                            <div style={{ opacity: 0.7, marginTop: '4px' }}>
                                                Reasoning: {msg.reasoning}
                                            </div>
                                        )}
                                    </div>
                                )}
                            </motion.div>
                        ))}
                        {loading && (
                            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="message system">
                                Thinking...
                            </motion.div>
                        )}
                    </AnimatePresence>
                    <div ref={chatEndRef} />
                </div>

                <div className="model-selector-container">
                    <div
                        className="model-selector-header"
                        onClick={() => setShowModels(!showModels)}
                        style={{
                            cursor: 'pointer',
                            display: 'flex',
                            justifyContent: 'space-between',
                            background: 'rgba(255, 255, 255, 0.05)',
                            padding: '8px 12px',
                            borderRadius: '8px',
                            marginBottom: '10px'
                        }}
                    >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <Activity size={12} style={{ color: 'var(--groq-orange)' }} />
                            <span style={{ color: 'var(--text-primary)', fontWeight: '700' }}>
                                {docId ? documents.find(d => d.id === docId)?.name || 'Active Model' : 'Select RAG Model'}
                            </span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <span style={{ fontSize: '10px', opacity: 0.5 }}>{showModels ? 'CLOSE' : 'CHANGE'}</span>
                            {showModels ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
                        </div>
                    </div>

                    <AnimatePresence>
                        {showModels && (
                            <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: 'auto', opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                style={{ overflow: 'hidden' }}
                            >
                                <div className="model-selector-tray">
                                    {documents.length === 0 ? (
                                        <div className="model-card-mini" style={{ opacity: 0.6, cursor: 'default', borderStyle: 'dashed' }}>
                                            <div className="model-status-dot"></div>
                                            <p>Upload a document to train your first model</p>
                                        </div>
                                    ) : (
                                        documents.map(doc => (
                                            <div
                                                key={doc.id}
                                                className={`model-card-mini ${docId === doc.id ? 'active' : ''}`}
                                                onClick={() => {
                                                    if (docId === doc.id) {
                                                        // Already active - just reset view to short message as requested
                                                        setMessages([{ role: 'system', content: `Active: ${doc.name}.` }]);
                                                    } else {
                                                        setDocId(doc.id);
                                                        setFile({ name: doc.name });
                                                    }
                                                    setShowModels(false); // Auto close after selection
                                                }}
                                            >
                                                <div className="model-status-dot"></div>
                                                <p>{doc.name}</p>
                                            </div>
                                        ))
                                    )}
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>

                <form className="input-area" onSubmit={handleQuery}>
                    <input
                        type="text"
                        placeholder={docId ? "Ask a question..." : "Select a RAG model above to start..."}
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        disabled={!docId || loading}
                    />
                    <button type="submit" disabled={!docId || loading}>
                        <Send size={18} />
                    </button>
                </form>
            </div>
        </>
    );
};

export default ChatPage;
