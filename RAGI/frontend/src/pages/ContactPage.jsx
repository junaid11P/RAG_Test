import React from 'react';
import { Mail, MessageCircle, HelpCircle, Globe, Github, Linkedin } from 'lucide-react';

const ContactPage = ({ setView }) => {
    return (
        <div className="upgrade-page">
            <div className="upgrade-header">
                <h1 style={{ fontSize: '32px', fontWeight: '800', marginBottom: '8px' }}>Contact Support</h1>
                <p style={{ color: 'var(--text-secondary)', fontSize: '16px' }}>
                    We're here to help you build better RAG pipelines.
                </p>
            </div>

            <div className="pricing-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px' }}>
                {/* Email Support */}
                <div className="plan-card" style={{ textAlign: 'center', alignItems: 'center' }}>
                    <div style={{
                        width: '48px',
                        height: '48px',
                        borderRadius: '12px',
                        background: 'rgba(242, 101, 34, 0.1)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        marginBottom: '20px',
                        color: 'var(--groq-orange)'
                    }}>
                        <Mail size={24} />
                    </div>
                    <h3 style={{ fontSize: '18px', fontWeight: '800', marginBottom: '12px' }}>Email Us</h3>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginBottom: '24px', lineHeight: '1.6' }}>
                        For technical issues, billing inquiries, or general questions.
                    </p>
                    <a
                        href="mailto:junedjnr9632@gmail.com"
                        className="primary-btn"
                        style={{ textDecoration: 'none', width: '100%', textAlign: 'center', padding: '12px' }}
                    >
                        junedjnr9632@gmail.com
                    </a>
                </div>

                {/* Community & Docs */}
                <div className="plan-card" style={{ textAlign: 'center', alignItems: 'center' }}>
                    <div style={{
                        width: '48px',
                        height: '48px',
                        borderRadius: '12px',
                        background: 'rgba(139, 92, 246, 0.1)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        marginBottom: '20px',
                        color: '#8b5cf6'
                    }}>
                        <HelpCircle size={24} />
                    </div>
                    <h3 style={{ fontSize: '18px', fontWeight: '800', marginBottom: '12px' }}>Help Center</h3>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginBottom: '24px', lineHeight: '1.6' }}>
                        Browse our documentation and community forums for quick answers.
                    </p>
                    <button
                        className="btn-upgrade"
                        style={{ width: '100%' }}
                        onClick={() => setView('docs')}
                    >
                        Visit Docs
                    </button>
                </div>

                {/* Feedback */}
                <div className="plan-card" style={{ textAlign: 'center', alignItems: 'center' }}>
                    <div style={{
                        width: '48px',
                        height: '48px',
                        borderRadius: '12px',
                        background: 'rgba(0, 230, 118, 0.1)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        marginBottom: '20px',
                        color: '#00e676'
                    }}>
                        <MessageCircle size={24} />
                    </div>
                    <h3 style={{ fontSize: '18px', fontWeight: '800', marginBottom: '12px' }}>Feedback</h3>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginBottom: '24px', lineHeight: '1.6' }}>
                        Have a feature request or found a bug? We'd love to hear from you.
                    </p>
                    <button
                        className="btn-upgrade"
                        style={{ width: '100%' }}
                        onClick={() => window.location.href = 'mailto:junedjnr9632@gmail.com?subject=Feedback for RAGI'}
                    >
                        Give Feedback
                    </button>
                </div>
            </div>

            {/* Social Links */}
            <div style={{ marginTop: '60px', textAlign: 'center', borderTop: '1px solid var(--glass-border)', paddingTop: '40px' }}>
                <p style={{ color: 'var(--text-secondary)', fontSize: '12px', marginBottom: '20px', letterSpacing: '1px', fontWeight: '800' }}>FOLLOW US</p>
                <div style={{ display: 'flex', justifyContent: 'center', gap: '24px' }}>
                    <a href="https://github.com/junaid11P" target="_blank" rel="noopener noreferrer" className="theme-icon" style={{ opacity: 0.6 }}><Github size={20} /></a>
                    <a href="https://www.linkedin.com/in/juned11/" target="_blank" rel="noopener noreferrer" className="theme-icon" style={{ opacity: 0.6 }}><Linkedin size={20} /></a>
                    <a href="https://juned-portfolio.onrender.com/" target="_blank" rel="noopener noreferrer" className="theme-icon" style={{ opacity: 0.6 }}><Globe size={20} /></a>
                </div>
            </div>
        </div >
    );
};

export default ContactPage;
