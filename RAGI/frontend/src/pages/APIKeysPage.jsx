import React from 'react';
import { Copy, Pencil, Trash2, Plus } from 'lucide-react';
import CreateKeyModal from '../components/CreateKeyModal';

const APIKeysPage = ({
    documents,
    usage,
    generateKey,
    deleteDocument,
    confirmRegenId,
    setConfirmRegenId,
    showCreateModal,
    setShowCreateModal,
    selectedDocId,
    setSelectedDocId
}) => {
    return (
        <div className="api-keys-container">
            <div className="api-keys-header">
                <div>
                    <h1>API Keys</h1>
                    <p>Manage your project API keys. Remember to keep your API keys safe to prevent unauthorized access.</p>
                </div>
                <button className="primary-btn" onClick={() => setShowCreateModal(true)} style={{ padding: '10px 20px', gap: '8px', display: 'flex', alignItems: 'center' }}>
                    <Plus size={18} /> Create API Key
                </button>
            </div>

            <CreateKeyModal
                showCreateModal={showCreateModal}
                setShowCreateModal={setShowCreateModal}
                selectedDocId={selectedDocId}
                setSelectedDocId={setSelectedDocId}
                documents={documents}
                generateKey={generateKey}
            />

            <div className="api-keys-table-container">
                <table className="api-keys-table">
                    <thead>
                        <tr>
                            <th>NAME</th>
                            <th>SECRET KEY</th>
                            <th>CREATED</th>
                            <th>EXPIRES</th>
                            <th style={{ textAlign: 'right' }}>ACTIONS</th>
                        </tr>
                    </thead>
                    <tbody>
                        {documents.length === 0 ? (
                            <tr>
                                <td colSpan="5" className="empty-state">
                                    <div style={{ padding: '40px', opacity: 0.5 }}>
                                        No API keys found. Click "Create API Key" to get started.
                                    </div>
                                </td>
                            </tr>
                        ) : (
                            documents.map(doc => (
                                <tr key={doc.id}>
                                    <td className="doc-name-cell" data-label="NAME">
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', justifyContent: 'flex-end' }}>
                                            <div style={{
                                                width: '8px',
                                                height: '8px',
                                                borderRadius: '50%',
                                                background: doc.is_premium ? 'var(--groq-orange)' : '#8b949e'
                                            }}></div>
                                            <span style={{ fontWeight: '700', fontSize: '14px' }}>
                                                {doc.name.split('.')[0].toUpperCase()}
                                            </span>
                                        </div>
                                    </td>
                                    <td data-label="SECRET KEY">
                                        <div className="key-display-group" style={{ marginLeft: 'auto' }}>
                                            <code>
                                                {doc.api_key ? `${doc.api_key.slice(0, 8)}...${doc.api_key.slice(-4)}` : '—'}
                                            </code>
                                            {doc.api_key && (
                                                <button className="copy-btn-minimal" onClick={() => {
                                                    navigator.clipboard.writeText(doc.api_key);
                                                    alert('Key copied!');
                                                }}>
                                                    <Copy size={13} />
                                                </button>
                                            )}
                                        </div>
                                    </td>
                                    <td data-label="CREATED" style={{ color: '#8b949e', fontSize: '13px' }}>
                                        {new Date(doc.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}
                                    </td>
                                    <td data-label="EXPIRES">
                                        <span className={`badge ${doc.is_premium ? 'success' : 'warning'}`}>
                                            {doc.is_premium ? 'Permanent' : 'Trial (2d)'}
                                        </span>
                                    </td>
                                    <td data-label="ACTIONS" style={{ textAlign: 'right' }}>
                                        <div className="actions-cell">
                                            {confirmRegenId === doc.id ? (
                                                <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
                                                    <button className="btn-regen-active" onClick={() => {
                                                        generateKey(doc.id);
                                                        setConfirmRegenId(null);
                                                    }}>Confirm</button>
                                                    <button className="btn-cancel-minimal" onClick={() => setConfirmRegenId(null)}>Cancel</button>
                                                </div>
                                            ) : (
                                                <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
                                                    <button className="icon-btn-action"
                                                        onClick={() => setConfirmRegenId(doc.id)}
                                                        title="Regenerate Key">
                                                        <Pencil size={15} />
                                                    </button>
                                                    <button className="icon-btn-action delete"
                                                        onClick={() => deleteDocument(doc.id)}
                                                        title="Delete Key">
                                                        <Trash2 size={15} />
                                                    </button>
                                                </div>
                                            )}
                                        </div>
                                    </td>
                                </tr>
                            ))
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default APIKeysPage;
