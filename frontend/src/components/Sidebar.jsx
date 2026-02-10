import React from 'react';
import { Activity, Upload, FileText, History, ChevronRight } from 'lucide-react';

const Sidebar = ({ user, usage, uploading, file, handleFileUpload, documents, docId, setDocId, setFile }) => {
    return (
        <div className="sidebar">
            <div className="glass glass-card">
                <h3 className="sidebar-section-title">
                    <Activity size={18} style={{ marginRight: '8px' }} />
                    Usage & Billing
                </h3>
                {!user ? (
                    <p style={{ fontSize: '14px', opacity: 0.7 }}>Login to track usage and billing.</p>
                ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                        <div className="usage-row">
                            <span className="usage-label">Files</span>
                            <span className="usage-value">{usage?.files_uploaded || 0}</span>
                        </div>
                        <div className="usage-row">
                            <span className="usage-label">Queries</span>
                            <span className="usage-value">{usage?.api_calls || 0}</span>
                        </div>
                        <div className="bill-container usage-row">
                            <span style={{ fontWeight: '600', color: '#00e676' }}>Total Bill</span>
                            <span className="bill-amount">${usage?.estimated_bill || 0.00}</span>
                        </div>
                    </div>
                )}
            </div>

            <div className="glass glass-card">
                <h3 className="sidebar-section-title" style={{ color: 'var(--text-primary)' }}>
                    <Upload size={18} style={{ marginRight: '8px' }} />
                    Documents
                </h3>
                <label className="upload-zone">
                    <input type="file" onChange={handleFileUpload} accept=".pdf,.txt,.docx,.doc" style={{ display: 'none' }} />
                    <FileText size={40} style={{ opacity: 0.5 }} />
                    <p style={{ fontSize: '13px', fontWeight: '500' }}>{uploading ? 'Processing...' : file ? file.name : 'Click to Upload PDF, TXT, or Word'}</p>
                </label>
            </div>

            <div className="glass glass-card" style={{ flex: 1, overflow: 'hidden' }}>
                <h3 className="sidebar-section-title">
                    <History size={18} style={{ marginRight: '8px' }} />
                    Recent Library
                </h3>
                <div className="library-container">
                    {documents.length === 0 ? (
                        <p style={{ fontSize: '13px', opacity: 0.5, textAlign: 'center' }}>No documents saved.</p>
                    ) : (
                        documents.map(doc => (
                            <div key={doc.id} className={`doc-item ${docId === doc.id ? 'active' : ''}`} onClick={() => { setDocId(doc.id); setFile({ name: doc.name }) }}>
                                <div style={{ flex: 1, overflow: 'hidden' }}>
                                    <p style={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>{doc.name}</p>
                                </div>
                                <ChevronRight size={14} style={{ opacity: 0.3 }} />
                            </div>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
};

export default Sidebar;
