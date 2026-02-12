import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { ChevronLeft, Mail, QrCode, AlertCircle, Send, CheckCircle2, Loader2, History, Heart, Rocket } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

const PaymentPage = ({ setView, user }) => {
    const [utrNumber, setUtrNumber] = useState('');
    const [submitting, setSubmitting] = useState(false);
    const [verifications, setVerifications] = useState([]);
    const [submitted, setSubmitted] = useState(false);

    useEffect(() => {
        if (user) {
            fetchVerificationStatus();
        }
    }, [user]);

    const fetchVerificationStatus = async () => {
        try {
            const resp = await axios.get(`${API_BASE}/api/payments/status`, {
                headers: { Authorization: `Bearer ${user.token}` }
            });
            setVerifications(resp.data);
            const hasPending = resp.data.some(v => v.status === 'pending');
            if (hasPending) setSubmitted(true);
        } catch (e) {
            console.error("Failed to fetch support status", e);
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!utrNumber.trim()) return;

        setSubmitting(true);
        try {
            await axios.post(`${API_BASE}/api/payments/verify`, {
                utr_number: utrNumber,
                email: user.email,
                type: 'donation'
            }, {
                headers: { Authorization: `Bearer ${user.token}` }
            });
            setSubmitted(true);
            fetchVerificationStatus();
        } catch (error) {
            alert('Submission failed. Please try again.');
        } finally {
            setSubmitting(false);
        }
    };

    return (
        <div className="upgrade-page" style={{ padding: '0 20px 60px' }}>
            <div className="upgrade-header">
                <button
                    onClick={() => setView('upgrade')}
                    style={{
                        background: 'none',
                        border: 'none',
                        color: 'var(--text-secondary)',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                        cursor: 'pointer',
                        fontSize: '14px',
                        marginBottom: '20px',
                        padding: '0'
                    }}
                >
                    <ChevronLeft size={16} /> Back to Plans
                </button>

                {/* Coming Soon Banner */}
                <div style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '8px',
                    background: 'rgba(139, 92, 246, 0.1)',
                    padding: '8px 16px',
                    borderRadius: '100px',
                    color: 'var(--accent-color)',
                    fontSize: '12px',
                    fontWeight: '800',
                    textTransform: 'uppercase',
                    letterSpacing: '1px',
                    marginBottom: '20px'
                }}>
                    <Rocket size={14} /> Developer Pro is Coming Soon
                </div>

                <h1 style={{ fontSize: '32px', fontWeight: '800', marginBottom: '8px' }}>Support Our Journey</h1>
                <p style={{ color: 'var(--text-secondary)', fontSize: '16px', maxWidth: '600px' }}>
                    If you appreciate RAGI, consider supporting us with a small donation.
                    Your contribution helps us build the future of intelligent document chat.
                </p>
            </div>

            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
                gap: '32px',
                maxWidth: '1000px',
                margin: '0 auto'
            }}>
                {/* QR Code Card */}
                <div style={{
                    padding: '40px',
                    background: 'var(--glass-bg)',
                    backdropFilter: 'blur(12px)',
                    borderRadius: '24px',
                    border: '1px solid var(--glass-border)',
                    textAlign: 'center'
                }}>
                    <div style={{ marginBottom: '24px' }}>
                        <div style={{
                            width: '48px',
                            height: '48px',
                            borderRadius: '12px',
                            background: 'rgba(236, 72, 153, 0.1)',
                            color: '#ec4899',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            margin: '0 auto 16px'
                        }}>
                            <Heart size={24} fill="currentColor" />
                        </div>
                        <h3 style={{ fontSize: '18px', fontWeight: '800' }}>Appreciation QR</h3>
                    </div>

                    <div style={{
                        padding: '16px',
                        background: 'white',
                        borderRadius: '16px',
                        boxShadow: '0 8px 32px rgba(0,0,0,0.2)',
                        width: 'fit-content',
                        margin: '0 auto 24px'
                    }}>
                        <img
                            src="/QR_Pay.jpg"
                            alt="Donation QR Code"
                            style={{ width: '200px', height: '200px', display: 'block', borderRadius: '8px' }}
                        />
                    </div>

                    <p style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
                        Scan using any UPI app to support us.<br />
                        Any small amount is deeply appreciated!
                    </p>
                </div>

                {/* Verification Form Card */}
                <div style={{
                    padding: '40px',
                    background: 'var(--glass-bg)',
                    backdropFilter: 'blur(12px)',
                    borderRadius: '24px',
                    border: '1px solid var(--glass-border)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '24px'
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', color: 'var(--groq-orange)' }}>
                        <AlertCircle size={20} />
                        <h3 style={{ fontSize: '18px', fontWeight: '800', color: 'var(--text-primary)' }}>Donation Support</h3>
                    </div>

                    {!submitted ? (
                        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                <label style={{ fontSize: '12px', fontWeight: '800', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
                                    UTR / Transaction ID (Optional)
                                </label>
                                <input
                                    type="text"
                                    placeholder="Enter UTR Number"
                                    value={utrNumber}
                                    onChange={(e) => setUtrNumber(e.target.value)}
                                    style={{
                                        background: 'rgba(0,0,0,0.2)',
                                        border: '1px solid var(--glass-border)',
                                        padding: '12px 16px',
                                        borderRadius: '12px',
                                        color: 'white'
                                    }}
                                />
                            </div>

                            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
                                Account Email: <strong style={{ color: 'white' }}>{user?.email}</strong>
                            </p>

                            <button
                                type="submit"
                                disabled={submitting}
                                className="primary-btn"
                                style={{
                                    width: '100%',
                                    padding: '14px',
                                    borderRadius: '12px',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    gap: '10px'
                                }}
                            >
                                {submitting ? <Loader2 className="animate-spin" size={20} /> : <Heart size={20} />}
                                Send Thanks
                            </button>

                            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', textAlign: 'center' }}>
                                We'll keep your account in our priority list for the Pro release!
                            </p>
                        </form>
                    ) : (
                        <div style={{ textAlign: 'center', padding: '20px 0', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
                            <div style={{ color: '#ec4899' }}>
                                <Heart size={64} fill="currentColor" />
                            </div>
                            <div>
                                <h4 style={{ fontSize: '18px', fontWeight: '800', marginBottom: '8px' }}>Thank You So Much!</h4>
                                <p style={{ fontSize: '14px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
                                    Your support means the world to us. We've noted your contribution and will notify you when Pro is ready!
                                </p>
                            </div>
                            <button
                                onClick={() => setSubmitted(false)}
                                style={{ background: 'none', border: 'none', color: 'var(--accent-color)', fontSize: '13px', cursor: 'pointer', fontWeight: '600' }}
                            >
                                Submit another note?
                            </button>
                        </div>
                    )}
                </div>
            </div>

            {/* History Section */}
            {verifications.length > 0 && (
                <div style={{ maxWidth: '1000px', margin: '60px auto 0' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '24px', color: 'var(--text-secondary)' }}>
                        <History size={18} />
                        <h3 style={{ fontSize: '16px', fontWeight: '700' }}>Your Support History</h3>
                    </div>
                    <div style={{ background: 'var(--glass-bg)', borderRadius: '16px', overflow: 'hidden', border: '1px solid var(--glass-border)' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
                            <thead>
                                <tr style={{ background: 'rgba(255,255,255,0.02)' }}>
                                    <th style={{ padding: '16px' }}>Transaction Info</th>
                                    <th style={{ padding: '16px' }}>Date</th>
                                    <th style={{ padding: '16px' }}>Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                {verifications.map((v) => (
                                    <tr key={v.id} style={{ borderTop: '1px solid var(--glass-border)' }}>
                                        <td style={{ padding: '16px', fontFamily: 'monospace' }}>{v.utr_number || 'Gratitude Note'}</td>
                                        <td style={{ padding: '16px', color: 'var(--text-secondary)' }}>
                                            {new Date(v.created_at).toLocaleDateString()}
                                        </td>
                                        <td style={{ padding: '16px' }}>
                                            <span style={{
                                                padding: '4px 8px',
                                                borderRadius: '6px',
                                                fontSize: '11px',
                                                fontWeight: '800',
                                                textTransform: 'uppercase',
                                                background: 'rgba(236, 72, 153, 0.1)',
                                                color: '#ec4899'
                                            }}>
                                                Grateful
                                            </span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
};

export default PaymentPage;
