import React from 'react';

const CreateKeyModal = ({ showCreateModal, setShowCreateModal, selectedDocId, setSelectedDocId, documents, generateKey }) => {
    if (!showCreateModal) return null;

    return (
        <div className="modal-overlay" onClick={() => setShowCreateModal(false)}>
            <div className="glass modal-card" onClick={e => e.stopPropagation()} style={{ maxWidth: '460px' }}>
                <div className="modal-header" style={{ marginBottom: '20px' }}>
                    <h2 style={{ margin: 0, fontSize: '20px', fontWeight: '800' }}>New API Key</h2>
                </div>

                <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginBottom: '24px', lineHeight: '1.6' }}>
                    Generate a unique access key for your document. This key will allow you to query the document content programmatically.
                </p>

                <div className="input-group" style={{ marginBottom: '32px' }}>
                    <label style={{ fontSize: '11px', fontWeight: '800', letterSpacing: '1px', marginBottom: '12px', display: 'block' }}>
                        SELECT SOURCE DOCUMENT
                    </label>
                    <select
                        value={selectedDocId}
                        onChange={(e) => setSelectedDocId(e.target.value)}
                        className="glass-select"
                        style={{
                            width: '100%',
                            padding: '12px',
                            borderRadius: '8px',
                            background: 'rgba(255,255,255,0.05)',
                            border: '1px solid var(--glass-border)',
                            color: 'white',
                            outline: 'none',
                            cursor: 'pointer'
                        }}
                    >
                        <option value="" disabled style={{ background: '#0d0b1a' }}>Choose from library...</option>
                        {documents.filter(d => !d.api_key).map(doc => (
                            <option key={doc.id} value={doc.id} style={{ background: '#0d0b1a' }}>
                                {doc.name.toUpperCase()}
                            </option>
                        ))}
                    </select>
                    {documents.filter(d => !d.api_key).length === 0 && (
                        <p style={{ fontSize: '12px', color: '#f85149', marginTop: '10px', fontWeight: '500' }}>
                            ⚠️ All documents already have keys. Upload new ones in chat!
                        </p>
                    )}
                </div>

                <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
                    <button
                        className="btn-cancel-minimal"
                        style={{ padding: '10px 20px' }}
                        onClick={() => {
                            setShowCreateModal(false);
                            setSelectedDocId('');
                        }}
                    >
                        Discard
                    </button>
                    <button
                        className="primary-btn"
                        disabled={!selectedDocId}
                        onClick={async () => {
                            await generateKey(selectedDocId);
                            setShowCreateModal(false);
                            setSelectedDocId('');
                        }}
                        style={{
                            padding: '10px 24px',
                            fontSize: '14px',
                            opacity: !selectedDocId ? 0.5 : 1,
                            cursor: !selectedDocId ? 'not-allowed' : 'pointer'
                        }}
                    >
                        Create Key
                    </button>
                </div>
            </div>
        </div>
    );
};

export default CreateKeyModal;
