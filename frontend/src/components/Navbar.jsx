import React, { useState, useRef, useEffect } from 'react';
import { LogOut, User, Sun, Moon, Monitor, Zap, Activity, MessageSquare, MessageCircle, Cookie, ChevronDown, Contact, Brain, BookOpen, Menu, X } from 'lucide-react';

const Navbar = ({ user, view, setView, logout, setShowAuth }) => {
    const [showProfileMenu, setShowProfileMenu] = useState(false);
    const [showMobileMenu, setShowMobileMenu] = useState(false);
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

    const navItems = [
        { id: 'chat', label: 'Chat', show: true },
        { id: 'dev', label: 'API', show: !!user },
        { id: 'docs', label: 'Docs', show: true },
        { id: 'upgrade', label: 'Upgrade', show: true },
        { id: 'contact', label: 'Contact', show: true },
    ];

    return (
        <nav className="glass">
            <div
                className="logo"
                onClick={() => setView('chat')}
                style={{
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    transition: 'all 0.3s ease'
                }}
            >
                <div style={{
                    background: 'var(--gradient)',
                    padding: '6px',
                    borderRadius: '10px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 4px 12px rgba(139, 92, 246, 0.2)'
                }}>
                    <Brain size={20} color="white" />
                </div>
                RAGI
            </div>

            {/* Desktop Nav */}
            <div className="desktop-nav" style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <div className="nav-links" style={{ display: 'flex', gap: '8px' }}>
                    {navItems.filter(item => item.show).map(item => (
                        <button
                            key={item.id}
                            className={`nav-btn ${view === item.id ? 'active' : ''}`}
                            onClick={() => setView(item.id)}
                        >
                            {item.label}
                        </button>
                    ))}
                </div>
            </div>

            <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
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
                                border: '1px solid var(--glass-border)',
                                cursor: 'pointer'
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
                            <span className="desktop-only" style={{ fontSize: '13px', fontWeight: '600' }}>{userName}</span>
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

                                    <button className="menu-item" onClick={() => { setView('docs'); setShowProfileMenu(false); }}>
                                        <BookOpen className="menu-icon" size={16} />
                                        <span>Docs</span>
                                    </button>

                                    <button className="menu-item" onClick={() => { setView('contact'); setShowProfileMenu(false); }}>
                                        <Contact className="menu-icon contact-icon" size={16} />
                                        <span>Contact</span>
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
                    <button className="primary-btn desktop-only" style={{ padding: '10px 24px' }} onClick={() => setShowAuth(true)}>Login</button>
                )}

                {/* Hamburger Overlay Button */}
                <button
                    className="mobile-only glass"
                    onClick={() => setShowMobileMenu(true)}
                    style={{
                        padding: '10px',
                        display: 'none', // Managed via CSS
                        borderRadius: '12px',
                        background: 'var(--glass-bg)',
                        border: '1px solid var(--glass-border)'
                    }}
                >
                    <Menu size={20} />
                </button>
            </div>

            {/* Mobile Menu Overlay */}
            {showMobileMenu && (
                <div className="mobile-menu-overlay">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div className="logo" style={{ fontSize: '24px' }}>RAGI</div>
                        <button
                            onClick={() => setShowMobileMenu(false)}
                            style={{ background: 'none', border: 'none', color: 'white' }}
                        >
                            <X size={28} />
                        </button>
                    </div>

                    <div className="mobile-menu-links">
                        {navItems.filter(item => item.show).map(item => (
                            <button
                                key={item.id}
                                className={`mobile-nav-btn ${view === item.id ? 'active' : ''}`}
                                onClick={() => {
                                    setView(item.id);
                                    setShowMobileMenu(false);
                                }}
                            >
                                {item.label}
                            </button>
                        ))}
                        {!user && (
                            <button
                                className="mobile-nav-btn"
                                style={{ background: 'var(--gradient)' }}
                                onClick={() => {
                                    setShowAuth(true);
                                    setShowMobileMenu(false);
                                }}
                            >
                                Login / Signup
                            </button>
                        )}
                    </div>

                    {user && (
                        <div style={{ marginTop: 'auto', borderTop: '1px solid var(--glass-border)', paddingTop: '24px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
                                <div style={{
                                    width: '48px',
                                    height: '48px',
                                    borderRadius: '50%',
                                    background: 'var(--gradient)',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    fontSize: '20px',
                                    fontWeight: '800'
                                }}>
                                    {userName.charAt(0).toUpperCase()}
                                </div>
                                <div>
                                    <div style={{ fontWeight: '700' }}>{userName}</div>
                                    <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{user.email}</div>
                                </div>
                            </div>
                            <button
                                className="mobile-nav-btn"
                                style={{ color: '#ff4444', borderColor: 'rgba(255, 68, 68, 0.2)' }}
                                onClick={() => {
                                    logout();
                                    setShowMobileMenu(false);
                                }}
                            >
                                <LogOut size={18} style={{ marginRight: '8px' }} />
                                Logout
                            </button>
                        </div>
                    )}
                </div>
            )}
        </nav>
    );
};

export default Navbar;
