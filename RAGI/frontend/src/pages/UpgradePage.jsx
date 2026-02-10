import React from 'react';
import { Check, Zap, ExternalLink } from 'lucide-react';

const UpgradePage = ({ setView, user, setShowAuth }) => {
    return (
        <div className="upgrade-page">
            <div className="upgrade-header">
                <h1>Billing</h1>
                <div className="upgrade-tabs">
                    <div className="upgrade-tab active">Plans</div>
                </div>
            </div>

            <div className="pricing-grid">
                {/* Free Plan */}
                <div className="plan-card free highlight">
                    <div className="plan-type">
                        <Zap size={16} fill="currentColor" /> Free / Trial
                    </div>
                    <p className="plan-desc">Great for getting started and testing your RAG pipelines</p>
                    <div className="plan-price">
                        <span className="price-amount">$0</span>
                    </div>
                    <button className="plan-btn btn-current">Current Plan</button>

                    <div className="feature-list">
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>2-Day Document Retention</span>
                        </div>
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>Per-Document API Keys</span>
                        </div>
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>Basic Vector Indexing</span>
                        </div>
                    </div>
                </div>

                {/* Developer Plan */}
                <div className="plan-card dev">
                    <div className="plan-type" style={{ color: 'var(--groq-orange)' }}>Developer Pro</div>
                    <p className="plan-desc">Perfect for scaling applications with permanent document context</p>
                    <div className="plan-price">
                        <span className="price-unit">Usage Based</span>
                    </div>
                    <button
                        className="plan-btn btn-upgrade"
                        onClick={() => {
                            if (user) {
                                window.open('https://stripe.com', '_blank'); // Placeholder for actual upgrade
                            } else {
                                setShowAuth(true);
                            }
                        }}
                    >
                        {user ? 'Upgrade Now' : 'Login to Upgrade'}
                    </button>
                    <p className="sub-text">
                        $0.10 per MB • $0.01 per Query. <a href="#">Billing FAQ</a>
                    </p>

                    <div className="feature-list">
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>Permanent Document Storage</span>
                        </div>
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>Unlimited API Keys</span>
                        </div>
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>Premium Support Tier</span>
                        </div>
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>Full Conversation History</span>
                        </div>
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>Custom Spend Limits</span>
                        </div>
                    </div>
                </div>

                {/* Enterprise Plan */}
                <div className="plan-card enterprise">
                    <div className="plan-type" style={{ color: 'var(--groq-orange)' }}>Enterprise</div>
                    <p className="plan-desc">Advanced security and scalability for high-volume enterprise needs</p>
                    <div className="plan-price" style={{ minHeight: '48px' }}></div>
                    <button className="plan-btn btn-contact" onClick={() => setView('contact')}>Contact Sales</button>

                    <div className="feature-list">
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>Everything in Pro, Plus:</span>
                        </div>
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>Dedicated Vector Clusters</span>
                        </div>
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>SLA-backed Uptime</span>
                        </div>
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>SSO & Team Management</span>
                        </div>
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>Custom Data Governance</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default UpgradePage;
