import React from 'react';
import { Check, Zap, Shield, Rocket, Globe } from 'lucide-react';

const UpgradePage = ({ setView, user, setShowAuth }) => {
    return (
        <div className="upgrade-page" style={{ padding: '0 20px 40px' }}>
            <div className="upgrade-header" style={{ textAlign: 'center', marginBottom: '60px' }}>
                <h1 style={{
                    fontSize: '48px',
                    fontWeight: '800',
                    marginBottom: '16px',
                    background: 'var(--gradient)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent'
                }}>
                    Elevate Your RAG Experience
                </h1>
                <p style={{ color: 'var(--text-secondary)', fontSize: '18px', maxWidth: '600px', margin: '0 auto' }}>
                    Unlock permanent storage, unlimited API access, and enterprise-grade security.
                </p>
            </div>

            <div className="pricing-grid" style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
                gap: '32px',
                maxWidth: '1200px',
                margin: '0 auto'
            }}>
                {/* Free Plan */}
                <div className="plan-card" style={{
                    position: 'relative',
                    background: 'var(--glass-bg)',
                    borderRadius: '24px',
                    padding: '40px',
                    border: '1px solid var(--glass-border)',
                    transition: 'all 0.3s ease',
                    display: 'flex',
                    flexDirection: 'column'
                }}>
                    <div style={{ marginBottom: '32px' }}>
                        <div style={{ color: 'var(--text-secondary)', fontSize: '14px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '2px', marginBottom: '8px' }}>Essential</div>
                        <h2 style={{ fontSize: '28px', fontWeight: '800' }}>Free Trial</h2>
                    </div>

                    <div style={{ marginBottom: '32px' }}>
                        <span style={{ fontSize: '48px', fontWeight: '800' }}>$0</span>
                        <span style={{ color: 'var(--text-secondary)' }}>/forever</span>
                    </div>

                    <button className="plan-btn" style={{
                        width: '100%',
                        padding: '16px',
                        borderRadius: '12px',
                        background: 'rgba(255,255,255,0.05)',
                        color: 'white',
                        border: '1px solid var(--glass-border)',
                        fontWeight: '700',
                        marginBottom: '32px'
                    }}>
                        Current Plan
                    </button>

                    <div className="feature-list" style={{ display: 'flex', flexDirection: 'column', gap: '16px', flex: 1 }}>
                        <FeatureItem text="2-Day Document Retention" />
                        <FeatureItem text="Per-Document API Keys" />
                        <FeatureItem text="Standard Vector Indexing" />
                        <FeatureItem text="Community Support" />
                    </div>
                </div>

                {/* Developer Pro Plan */}
                <div className="plan-card" style={{
                    position: 'relative',
                    background: 'rgba(139, 92, 246, 0.05)',
                    borderRadius: '24px',
                    padding: '40px',
                    border: '2px solid var(--accent-color)',
                    boxShadow: '0 0 40px rgba(139, 92, 246, 0.15)',
                    transition: 'all 0.3s ease',
                    display: 'flex',
                    flexDirection: 'column',
                    transform: 'scale(1.05)',
                    zIndex: 2
                }}>
                    <div style={{
                        position: 'absolute',
                        top: '-15px',
                        left: '50%',
                        transform: 'translateX(-50%)',
                        background: 'var(--gradient)',
                        padding: '6px 16px',
                        borderRadius: '100px',
                        fontSize: '12px',
                        fontWeight: '800',
                        textTransform: 'uppercase',
                        letterSpacing: '1px'
                    }}>
                        Most Popular
                    </div>

                    <div style={{ marginBottom: '32px' }}>
                        <div style={{ color: 'var(--accent-color)', fontSize: '14px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '2px', marginBottom: '8px' }}>Professional</div>
                        <h2 style={{ fontSize: '28px', fontWeight: '800' }}>Developer Pro</h2>
                    </div>

                    <div style={{ marginBottom: '16px' }}>
                        <span style={{ fontSize: '48px', fontWeight: '800' }}>Pay-As-You-Go</span>
                    </div>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginBottom: '32px' }}>
                        $0.10/MB • $0.01/Query
                    </p>

                    <button
                        className="primary-btn"
                        style={{
                            width: '100%',
                            padding: '16px',
                            borderRadius: '12px',
                            fontWeight: '700',
                            marginBottom: '32px',
                            cursor: 'pointer'
                        }}
                        onClick={() => user ? setView('payment') : setShowAuth(true)}
                    >
                        {user ? 'Upgrade Now' : 'Login to Upgrade'}
                    </button>

                    <div className="feature-list" style={{ display: 'flex', flexDirection: 'column', gap: '16px', flex: 1 }}>
                        <FeatureItem text="Permanent Document Storage" highlight />
                        <FeatureItem text="Global API Keys (Full Account)" highlight />
                        <FeatureItem text="Priority Vector Processing" highlight />
                        <FeatureItem text="Full Conversation History" highlight />
                        <FeatureItem text="Custom Usage Limits" highlight />
                        <FeatureItem text="Email Support" highlight />
                    </div>
                </div>

                {/* Enterprise Plan */}
                <div className="plan-card" style={{
                    position: 'relative',
                    background: 'var(--glass-bg)',
                    borderRadius: '24px',
                    padding: '40px',
                    border: '1px solid var(--glass-border)',
                    transition: 'all 0.3s ease',
                    display: 'flex',
                    flexDirection: 'column'
                }}>
                    <div style={{ marginBottom: '32px' }}>
                        <div style={{ color: 'var(--text-secondary)', fontSize: '14px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '2px', marginBottom: '8px' }}>Scalable</div>
                        <h2 style={{ fontSize: '28px', fontWeight: '800' }}>Enterprise</h2>
                    </div>

                    <div style={{ marginBottom: '32px' }}>
                        <span style={{ fontSize: '48px', fontWeight: '800' }}>Custom</span>
                    </div>

                    <button
                        className="plan-btn"
                        style={{
                            width: '100%',
                            padding: '16px',
                            borderRadius: '12px',
                            background: 'rgba(255,255,255,0.05)',
                            color: 'white',
                            border: '1px solid var(--glass-border)',
                            fontWeight: '700',
                            marginBottom: '32px',
                            cursor: 'pointer'
                        }}
                        onClick={() => setView('contact')}
                    >
                        Contact Sales
                    </button>

                    <div className="feature-list" style={{ display: 'flex', flexDirection: 'column', gap: '16px', flex: 1 }}>
                        <FeatureItem text="Everything in Pro" />
                        <FeatureItem text="Dedicated Compute Clusters" />
                        <FeatureItem text="SLA & Uptime Guarantees" />
                        <FeatureItem text="Custom Governance & SSO" />
                        <FeatureItem text="24/7 Dedicated Support" />
                    </div>
                </div>
            </div>
        </div>
    );
};

const FeatureItem = ({ text, highlight }) => (
    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '14px' }}>
        <div style={{
            width: '20px',
            height: '20px',
            borderRadius: '50%',
            background: highlight ? 'var(--accent-color)' : 'rgba(255,255,255,0.1)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0
        }}>
            <Check size={12} color="white" strokeWidth={3} />
        </div>
        <span style={{ color: highlight ? 'var(--text-primary)' : 'var(--text-secondary)' }}>{text}</span>
    </div>
);

export default UpgradePage;
