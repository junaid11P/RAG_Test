import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, ShieldCheck } from 'lucide-react';

const AuthModal = ({ showAuth, setShowAuth, authMode, setAuthMode, email, setEmail, password, setPassword, handleAuth }) => {
    return (
        <AnimatePresence>
            {showAuth && (
                <div className="modal-overlay" onClick={() => setShowAuth(false)}>
                    <motion.div
                        className="glass glass-card modal-card"
                        style={{ maxWidth: '440px', width: '90%', margin: '0 auto' }}
                        onClick={e => e.stopPropagation()}
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0.8, opacity: 0 }}
                    >
                        <div className="modal-header">
                            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                <div className="icon-badge">
                                    <ShieldCheck size={20} color="var(--accent-color)" />
                                </div>
                                <h3>{authMode === 'login' ? 'Welcome Back' : 'Create Account'}</h3>
                            </div>
                            <button className="close-btn" onClick={() => setShowAuth(false)}>
                                <X size={18} />
                            </button>
                        </div>

                        <form onSubmit={handleAuth} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                            <div className="input-group">
                                <label>EMAIL ADDRESS</label>
                                <input
                                    type="email"
                                    placeholder="name@company.com"
                                    value={email}
                                    onChange={e => setEmail(e.target.value)}
                                    required
                                />
                            </div>
                            <div className="input-group">
                                <label>PASSWORD</label>
                                <input
                                    type="password"
                                    placeholder="••••••••"
                                    value={password}
                                    onChange={e => setPassword(e.target.value)}
                                    required
                                />
                            </div>

                            <button type="submit" className="primary-btn">
                                {authMode === 'login' ? 'Sign In to Portal' : 'Create Admin Account'}
                            </button>

                            <div style={{ textAlign: 'center', marginTop: '10px' }}>
                                <button
                                    type="button"
                                    onClick={() => setAuthMode(authMode === 'login' ? 'register' : 'login')}
                                    style={{ background: 'none', border: 'none', color: 'var(--accent-color)', fontSize: '13px', cursor: 'pointer' }}
                                >
                                    {authMode === 'login' ? "Don't have an account? Sign up" : "Already have an account? Log in"}
                                </button>
                            </div>
                        </form>
                    </motion.div>
                </div>
            )}
        </AnimatePresence>
    );
};

export default AuthModal;
