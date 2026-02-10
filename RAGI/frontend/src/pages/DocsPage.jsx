import React, { useState, useRef, useEffect } from 'react';
import { BookOpen, Code, Terminal, Copy, Check, ExternalLink, ChevronDown, Cpu, ShieldCheck, Zap, Server, Globe } from 'lucide-react';

const DocsPage = ({ setView, user, setShowAuth }) => {
    const [activeLang, setActiveLang] = useState('python');
    const [showLangMenu, setShowLangMenu] = useState(false);
    const [copied, setCopied] = useState(false);
    const menuRef = useRef(null);

    const languages = [
        { id: 'python', name: 'Python' },
        { id: 'javascript', name: 'JavaScript' },
        { id: 'curl', name: 'curl' }
    ];

    const codeSnippets = {
        python: `import requests

# 1. Your unique per-document API key from the Dashboard
API_KEY = "YOUR_RAGI_API_KEY"
# 2. Base URL of your RAGI instance
BASE_URL = "http://localhost:8000" 

def ask_ragi(question):
    """
    Query the internal knowledge base of your document.
    """
    url = f"{BASE_URL}/api/v1/query"
    headers = {"X-API-Key": API_KEY}
    params = {"query": question}
    
    try:
        # We send a POST request with the query as a parameter
        response = requests.post(url, headers=headers, params=params)
        response.raise_for_status()
        
        # The API returns JSON with 'query' and 'answer' keys
        data = response.json()
        return data.get("answer")
        
    except requests.exceptions.RequestException as e:
        return f"API Error: {str(e)}"

# Example: Asking a question
if __name__ == "__main__":
    result = ask_ragi("What are the key technical details?")
    print(f"RAGI Answer: {result}")`,
        javascript: `// Integration using the popular 'axios' library
const axios = require('axios');

// 1. Set your Contextual API Key
const API_KEY = 'YOUR_RAGI_API_KEY';
const BASE_URL = 'http://localhost:8000';

async function queryRagi(question) {
    try {
        const endpoint = \`\${BASE_URL}/api/v1/query\`;
        
        const response = await axios.post(endpoint, null, {
            headers: { 'X-API-Key': API_KEY },
            params: { query: question }
        });
        
        return response.data.answer;
        
    } catch (error) {
        console.error('RAGI API Error:', error.message);
        return null;
    }
}

// Example usage
queryRagi('Give me a summary of this document')
    .then(answer => console.log('Response:', answer));`,
        curl: `curl -X POST "http://localhost:8000/api/v1/query?query=What+is+RAGI" \\
     -H "X-API-Key: YOUR_RAGI_API_KEY" \\
     -H "Accept: application/json"`
    };

    const handleCopy = () => {
        navigator.clipboard.writeText(codeSnippets[activeLang]);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (menuRef.current && !menuRef.current.contains(event.target)) {
                setShowLangMenu(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const CodePanel = ({ code, language }) => {
        const lines = code.split('\n');
        const isTerminal = language === 'curl';

        return (
            <div style={{ position: 'relative' }}>
                <pre style={{
                    margin: '0',
                    padding: isTerminal ? '24px' : '24px 0',
                    background: 'transparent',
                    overflowX: 'auto',
                    fontFamily: '"JetBrains Mono", monospace',
                    fontSize: '12px',
                    lineHeight: '1.8',
                    color: '#e6edf3',
                    display: 'flex'
                }}>
                    {!isTerminal && (
                        <div style={{
                            padding: '0 16px',
                            textAlign: 'right',
                            color: 'rgba(255,255,255,0.15)',
                            userSelect: 'none',
                            borderRight: '1px solid rgba(255,255,255,0.05)',
                            marginRight: '16px',
                            minWidth: '40px'
                        }}>
                            {lines.map((_, i) => (
                                <div key={i}>{i + 1}</div>
                            ))}
                        </div>
                    )}
                    <code style={{ flex: 1, paddingRight: '24px' }}>
                        {lines.map((line, i) => (
                            <div key={i} style={{
                                whiteSpace: 'pre',
                                color: line.trim().startsWith('#') || line.trim().startsWith('//') ? '#6a737d' : 'inherit'
                            }}>
                                {line || ' '}
                                {'\n'}
                            </div>
                        ))}
                    </code>
                </pre>
            </div>
        );
    };

    return (
        <div className="docs-container" style={{ padding: '40px', maxWidth: '1400px', margin: '0 auto', color: 'var(--text-primary)' }}>

            {/* Page Header */}
            <header style={{ marginBottom: '60px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
                    <div style={{ background: 'var(--gradient)', padding: '10px', borderRadius: '12px', color: 'white', boxShadow: '0 0 20px -5px var(--accent-glow)' }}>
                        <Globe size={24} />
                    </div>
                    <span style={{ fontSize: '12px', fontWeight: '900', color: 'var(--groq-orange)', letterSpacing: '2px' }}>V1.0 DOCUMENTATION</span>
                </div>
                <h1 style={{ fontSize: '48px', fontWeight: '900', marginBottom: '16px', letterSpacing: '-1.5px' }}>Developer Quickstart</h1>
                <p style={{ color: 'var(--text-secondary)', fontSize: '18px', maxWidth: '700px', lineHeight: '1.6' }}>
                    Integrate your trained RAG models into any production environment with our high-performance API.
                </p>
            </header>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 500px', gap: '50px', alignItems: 'start' }}>

                {/* Left Side: Steps & Reference */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '50px' }}>

                    {/* Step 1: Authentication */}
                    <section>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
                            <div style={{ background: 'rgba(242, 101, 34, 0.15)', color: 'var(--groq-orange)', padding: '8px', borderRadius: '8px' }}>
                                <ShieldCheck size={20} />
                            </div>
                            <h2 style={{ fontSize: '26px', fontWeight: '800', letterSpacing: '-0.5px' }}>1. Authentication</h2>
                        </div>
                        <div className="glass" style={{ padding: '32px', borderRadius: '24px', lineHeight: '1.8' }}>
                            <p style={{ marginBottom: '24px', fontSize: '16px', color: 'var(--text-secondary)' }}>
                                All API requests must include your unique <b>Contextual API Key</b> in the request header. This key authorizes access to a specific document's trained context.
                            </p>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                                <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
                                    <div style={{ width: '24px', height: '24px', background: 'var(--gradient)', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '11px', fontWeight: '900', color: 'white', flexShrink: 0, marginTop: '4px' }}>1</div>
                                    <div>
                                        <p style={{ fontWeight: '700', marginBottom: '4px' }}>Generate Secret Key</p>
                                        <p style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
                                            Visit the
                                            <button
                                                onClick={() => {
                                                    if (user) {
                                                        setView('dev');
                                                    } else {
                                                        setShowAuth(true);
                                                    }
                                                }}
                                                style={{
                                                    background: 'rgba(255,255,255,0.05)',
                                                    border: '1px solid var(--glass-border)',
                                                    color: 'white',
                                                    padding: '2px 10px',
                                                    borderRadius: '6px',
                                                    cursor: 'pointer',
                                                    fontSize: '12px',
                                                    fontWeight: '700',
                                                    margin: '0 4px'
                                                }}
                                            >
                                                API Dashboard
                                            </button>
                                            and click <b>Generate Key</b> for your model.
                                        </p>
                                    </div>
                                </div>
                                <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
                                    <div style={{ width: '24px', height: '24px', background: 'var(--gradient)', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '11px', fontWeight: '900', color: 'white', flexShrink: 0, marginTop: '4px' }}>2</div>
                                    <div>
                                        <p style={{ fontWeight: '700', marginBottom: '4px' }}>Secure Storage</p>
                                        <p style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
                                            Never expose your key in frontend code. Store it in <b>environment variables</b> (.env) on your server.
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </section>

                    {/* Step 2: API Reference */}
                    <section>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
                            <div style={{ background: 'rgba(139, 92, 246, 0.15)', color: '#8b5cf6', padding: '8px', borderRadius: '8px' }}>
                                <Server size={20} />
                            </div>
                            <h2 style={{ fontSize: '26px', fontWeight: '800', letterSpacing: '-0.5px' }}>2. API Reference</h2>
                        </div>
                        <div className="glass" style={{ padding: '32px', borderRadius: '24px' }}>
                            <div style={{ paddingBottom: '24px', borderBottom: '1px solid var(--glass-border)', marginBottom: '24px' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                                    <span style={{ background: '#00e676', color: '#000', padding: '4px 10px', borderRadius: '6px', fontSize: '11px', fontWeight: '900', letterSpacing: '1px' }}>POST</span>
                                    <code style={{ fontSize: '17px', fontWeight: '800', fontFamily: 'monospace' }}>/api/v1/query</code>
                                </div>
                                <p style={{ color: 'var(--text-secondary)', fontSize: '15px' }}>The primary endpoint for retrieving context-aware answers from your models.</p>
                            </div>

                            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                                <div>
                                    <h4 style={{ fontSize: '11px', fontWeight: '900', color: 'var(--groq-orange)', marginBottom: '16px', letterSpacing: '1.5px' }}>REQUIRED HEADERS</h4>
                                    <div className="glass" style={{ background: 'rgba(0,0,0,0.2)', padding: '16px', borderRadius: '14px', border: '1px solid var(--glass-border)' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                            <code style={{ color: '#8b5cf6', fontWeight: '800' }}>X-API-Key</code>
                                            <span style={{ fontSize: '10px', color: 'var(--text-secondary)', fontWeight: '800' }}>STRING</span>
                                        </div>
                                        <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>The unique bearer token for your document context.</p>
                                    </div>
                                </div>

                                <div>
                                    <h4 style={{ fontSize: '11px', fontWeight: '900', color: 'var(--groq-orange)', marginBottom: '16px', letterSpacing: '1.5px' }}>QUERY PARAMETERS</h4>
                                    <div className="glass" style={{ background: 'rgba(0,0,0,0.2)', padding: '16px', borderRadius: '14px', border: '1px solid var(--glass-border)' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                            <code style={{ color: '#00e676', fontWeight: '800' }}>query</code>
                                            <span style={{ fontSize: '10px', color: 'var(--text-secondary)', fontWeight: '800' }}>STRING</span>
                                        </div>
                                        <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>The natural language question sent to the RAG model.</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </section>
                </div>

                {/* Right Side: Sticky Code Panel */}
                <div style={{ position: 'sticky', top: '24px' }}>
                    <div className="glass" style={{
                        borderRadius: '24px',
                        overflow: 'hidden',
                        background: '#0d0b1a',
                        border: '1px solid var(--glass-border)',
                        boxShadow: '0 30px 60px -12px rgba(0, 0, 0, 0.6)'
                    }}>
                        {/* Panel Header */}
                        <div style={{
                            padding: '16px 20px',
                            borderBottom: '1px solid var(--glass-border)',
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            background: 'rgba(255,255,255,0.03)'
                        }}>
                            <div style={{ position: 'relative' }} ref={menuRef}>
                                <button
                                    onClick={() => setShowLangMenu(!showLangMenu)}
                                    style={{
                                        background: 'rgba(255,255,255,0.05)',
                                        border: '1px solid var(--glass-border)',
                                        color: 'white',
                                        padding: '7px 14px',
                                        borderRadius: '10px',
                                        fontSize: '13px',
                                        fontWeight: '800',
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '8px',
                                        cursor: 'pointer',
                                        transition: 'all 0.2s'
                                    }}
                                    className="hover-highlight"
                                >
                                    {activeLang === 'curl' ? 'curl' : activeLang.charAt(0).toUpperCase() + activeLang.slice(1)}
                                    <ChevronDown size={14} style={{ opacity: 0.5 }} />
                                </button>

                                {showLangMenu && (
                                    <div className="profile-dropdown" style={{
                                        top: '100%',
                                        left: '0',
                                        width: '180px',
                                        marginTop: '10px',
                                        background: '#151225',
                                        zIndex: 2000,
                                        boxShadow: '0 15px 40px rgba(0,0,0,0.6)',
                                        border: '1px solid var(--glass-border)',
                                        borderRadius: '14px'
                                    }}>
                                        <div className="menu-list" style={{ padding: '6px' }}>
                                            {languages.map(lang => (
                                                <button
                                                    key={lang.id}
                                                    className={`menu-item ${activeLang === lang.id ? 'active' : ''}`}
                                                    onClick={() => {
                                                        setActiveLang(lang.id);
                                                        setShowLangMenu(false);
                                                    }}
                                                    style={{ justifyContent: 'space-between', padding: '12px 14px', borderRadius: '8px' }}
                                                >
                                                    <span style={{ fontWeight: '700' }}>{lang.name}</span>
                                                    {activeLang === lang.id && <Check size={14} color="var(--groq-orange)" />}
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>

                            <button onClick={handleCopy} style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', padding: '10px', borderRadius: '8px', transition: 'all 0.2s' }} className="hover-highlight">
                                {copied ? <Check size={20} color="#00e676" /> : <Copy size={20} />}
                            </button>
                        </div>

                        {/* Code Display Area */}
                        <div style={{ minHeight: '400px', display: 'flex', flexDirection: 'column' }}>
                            <CodePanel code={codeSnippets[activeLang]} language={activeLang} />
                        </div>

                        {/* Panel Footer */}
                        <div style={{ padding: '14px 20px', background: 'rgba(0,0,0,0.25)', borderTop: '1px solid var(--glass-border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#00e676', boxShadow: '0 0 12px #00e676' }}></div>
                                <span style={{ fontSize: '10px', opacity: 0.6, letterSpacing: '1.2px', fontWeight: '900', color: 'var(--text-primary)' }}>RAGI CLOUD v1.0.4</span>
                            </div>
                            <span style={{ fontSize: '10px', opacity: 0.4, fontWeight: '700' }}>SSL SECURE</span>
                        </div>
                    </div>
                </div>

            </div>

            {/* Sub-Footer: Related Navigation */}
            <div style={{
                maxWidth: '100%',
                margin: '100px auto 40px',
                paddingTop: '60px',
                borderTop: '1px solid var(--glass-border)',
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
                gap: '30px'
            }}>
                <div className="glass-card" onClick={() => setView('upgrade')} style={{ padding: '30px', borderRadius: '24px', cursor: 'pointer', transition: 'all 0.3s' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                        <div style={{ background: 'rgba(139, 92, 246, 0.1)', padding: '8px', borderRadius: '10px' }}>
                            <Zap size={20} color="#8b5cf6" />
                        </div>
                        <h4 style={{ fontSize: '18px', fontWeight: '800' }}>Limits & Quota</h4>
                    </div>
                    <p style={{ fontSize: '14px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>Monitor your usage metrics and explore premium storage and rate-limit tiers.</p>
                </div>
                <div className="glass-card" onClick={() => setView('contact')} style={{ padding: '30px', borderRadius: '24px', cursor: 'pointer', transition: 'all 0.3s' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                        <div style={{ background: 'rgba(0, 230, 118, 0.1)', padding: '8px', borderRadius: '10px' }}>
                            <BookOpen size={20} color="#00e676" />
                        </div>
                        <h4 style={{ fontSize: '18px', fontWeight: '800' }}>Community Help</h4>
                    </div>
                    <p style={{ fontSize: '14px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>Access our developer community, FAQs, and human-to-human technical support.</p>
                </div>
            </div>
        </div>
    );
};

export default DocsPage;
