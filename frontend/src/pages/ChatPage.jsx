import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send } from 'lucide-react';
import Sidebar from '../components/Sidebar';

const ChatPage = ({
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
    loading,
    query,
    setQuery,
    handleQuery,
    chatEndRef
}) => {
    return (
        <>
            <Sidebar
                user={user}
                usage={usage}
                uploading={uploading}
                file={file}
                handleFileUpload={handleFileUpload}
                documents={documents}
                docId={docId}
                setDocId={setDocId}
                setFile={setFile}
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
                                {msg.content}
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

                <form className="input-area" onSubmit={handleQuery}>
                    <input
                        type="text"
                        placeholder={docId ? "Ask a question..." : "Upload a PDF first"}
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
