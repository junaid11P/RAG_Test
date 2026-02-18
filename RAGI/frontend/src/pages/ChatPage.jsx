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

    // Base URL for media assets
    const API_BASE_URL = apiBase;

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
                                    components={{
                                        img: ({ node, ...props }) => {
                                            // Handle relative proxy URLs
                                            const src = props.src.startsWith('/api/media/')
                                                ? `${API_BASE_URL}${props.src}`
                                                : props.src;
                                            return (
                                                <img
                                                    {...props}
                                                    src={src}
                                                    style={{
                                                        maxWidth: '100%',
                                                        borderRadius: '12px',
                                                        marginTop: '10px',
                                                        border: '1px solid var(--glass-border)'
                                                    }}
                                                />
                                            );
                                        }
                                    }}
                                >
                                    {msg.content}
                                </ReactMarkdown>
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
