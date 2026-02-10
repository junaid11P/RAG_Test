import React from 'react';
import { Check, Zap, ExternalLink } from 'lucide-react';

const UpgradePage = () => {
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
                        <Zap size={16} fill="currentColor" /> Free
                    </div>
                    <p className="plan-desc">Great for anyone to get started with our APIs</p>
                    <div className="plan-price">
                        <span className="price-amount">$0</span>
                    </div>
                    <button className="plan-btn btn-current">Current Plan</button>

                    <div className="feature-list">
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>Build and Test on Groq</span>
                        </div>
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>Community Support</span>
                        </div>
                    </div>
                </div>

                {/* Developer Plan */}
                <div className="plan-card dev">
                    <div className="plan-type" style={{ color: '#f26522' }}>Developer</div>
                    <p className="plan-desc">Great for developers and startups to scale up and pay as you go</p>
                    <div className="plan-price">
                        <span className="price-unit">Pay per Token</span>
                    </div>
                    <button className="plan-btn btn-upgrade">Upgrade</button>
                    <p className="sub-text">
                        No charge today. Billed for tokens at month-end. <a href="#">Learn more</a>
                    </p>

                    <div className="feature-list">
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>Higher Token Limits</span>
                        </div>
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>Chat Support</span>
                        </div>
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>Flex Service Tier <ExternalLink size={12} className="external-link" /></span>
                        </div>
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>Batch Processing <ExternalLink size={12} className="external-link" /></span>
                        </div>
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>Spend Limits <ExternalLink size={12} className="external-link" /></span>
                        </div>
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>Audit Logs (7-Day Retention)</span>
                        </div>
                    </div>
                </div>

                {/* Enterprise Plan */}
                <div className="plan-card enterprise">
                    <div className="plan-type" style={{ color: '#f26522' }}>Enterprise</div>
                    <p className="plan-desc">Great for businesses who require custom solutions for large scale needs</p>
                    <div className="plan-price" style={{ minHeight: '48px' }}></div>
                    <button className="plan-btn btn-contact">Contact Us</button>

                    <div className="feature-list">
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>Everything in Developer, Plus:</span>
                        </div>
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>Scalable Capacity</span>
                        </div>
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>Dedicated Support</span>
                        </div>
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>LoRA Inference</span>
                        </div>
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>SSO & SCIM</span>
                        </div>
                        <div className="feature-item">
                            <Check size={16} className="check-icon" />
                            <span>Audit Logs (90-Day Retention)</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default UpgradePage;
