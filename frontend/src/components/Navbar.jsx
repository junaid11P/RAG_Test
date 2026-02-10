import React, { useState, useRef, useEffect } from 'react';
import { LogOut, User, Sun, Moon, Monitor, Zap, Activity, MessageSquare, MessageCircle, Cookie, ChevronDown } from 'lucide-react';

const Navbar = ({ user, view, setView, logout, setShowAuth }) => {
    const [showProfileMenu, setShowProfileMenu] = useState(false);
    const menuRef = useRef(null);

    // Close menu when clicking outside
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (menuRef.current && !menuRef.current.contains(event.target)) {
                setShowProfileMenu(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const userEmail = user?.email || user?.id || 'Guest User';
    const userName = userEmail.split('@')[0];

    return (
        <nav className="glass">
            <div className="logo">RAGI</div>

            <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
                {user && (
                    <div className="nav-links">
                        <button
                            className={`nav-btn ${view === 'chat' ? 'active' : ''}`}
                            onClick={() => setView('chat')}>Chat</button>
                        <button
                            className={`nav-btn ${view === 'dev' ? 'active' : ''}`}
                            onClick={() => setView('dev')}>API</button>
                    </div>
                )}
            </div>

            <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
                {user ? (
                    <div className="profile-container" ref={menuRef}>
                        <div
                            className="glass"
                            style={{
                                display: 'flex',
                                gap: '10px',
                                alignItems: 'center',
                                padding: '6px 12px',
                                borderRadius: '100px',
                                border: '1px solid var(--glass-border)'
                            }}
                            onClick={() => setShowProfileMenu(!showProfileMenu)}
                        >
                            <div style={{
                                width: '32px',
                                height: '32px',
                                borderRadius: '50%',
                                background: 'var(--gradient)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                fontSize: '14px',
                                fontWeight: '800',
                                color: 'white',
                                boxShadow: '0 4px 12px rgba(139, 92, 246, 0.3)'
                            }}>
                                {userName.charAt(0).toUpperCase()}
                            </div>
                            {user?.email && (
                                <span style={{ fontSize: '13px', fontWeight: '600' }}>{userName}</span>
                            )}
                            <ChevronDown size={14} style={{ opacity: 0.5 }} />
                        </div>

                        {showProfileMenu && (
                            <div className="profile-dropdown">
                                <div className="profile-header">
                                    <div className="profile-user-info">
                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '4px' }}>
                                            <span className="profile-name" style={{
                                                fontSize: '15px',
                                                fontWeight: '800',
                                                color: 'white',
                                                wordBreak: 'break-all',
                                                maxWidth: '180px'
                                            }}>
                                                {user?.email ? user.email.split('@')[0] : 'User Account'}
                                            </span>
                                            <div className="theme-controls" style={{ display: 'flex', gap: '8px' }}>
                                                <Sun size={14} className="theme-icon" style={{ color: 'var(--groq-orange)' }} />
                                                <Moon size={14} className="theme-icon" />
                                                <Monitor size={14} className="theme-icon" />
                                            </div>
                                        </div>
                                        <span className="profile-email" style={{
                                            fontSize: '12px',
                                            color: 'var(--text-secondary)',
                                            display: 'block'
                                        }}>
                                            {user?.email || `ID: ${user?.id?.slice(0, 8)}...`}
                                        </span>
                                    </div>
                                </div>

                                <div className="menu-list">
                                    <button className="menu-item" onClick={() => { setView('upgrade'); setShowProfileMenu(false); }}>
                                        <Zap className="menu-icon upgrade-icon" size={16} />
                                        <span>Upgrade</span>
                                    </button>

                                    <button className="menu-item logout" onClick={logout}>
                                        <LogOut className="menu-icon" size={16} />
                                        <span>Log out</span>
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                ) : (
                    <button className="primary-btn" style={{ padding: '10px 24px' }} onClick={() => setShowAuth(true)}>Login / Signup</button>
                )}
            </div>
        </nav>
    );
};

export default Navbar;
