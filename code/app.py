"""
FastAPI Application for NCERT Physics RAG System (100% Error-Free Frontend)
Fixed regex escaping issues
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from pathlib import Path

from vector_db import MultimodalVectorDB
from rag_engine import LLMIntegratedRAGEngine, ResponseFormatter

app = FastAPI(
    title="NCERT Physics 11th Grade RAG System with LLM",
    description="Advanced RAG system with LLM integration",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Path("extracted_content").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="extracted_content"), name="static")

print("Initializing RAG system...")
db = MultimodalVectorDB()
engine = LLMIntegratedRAGEngine(db, "extracted_content/extracted_content.json")
formatter = ResponseFormatter()

class QueryRequest(BaseModel):
    query: str
    max_results: int = 10


def get_html_template():
    # Use raw string (r""") to avoid Python string escaping issues
    return r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NCERT Physics RAG with AI</title>
    <link rel="icon" href="data:,"> <!-- silences favicon 404 -->
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea, #764ba2);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        .header {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 2.4em; margin-bottom: 8px; }
        .header .ai-badge {
            background: rgba(255,255,255,0.25);
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 0.95em;
        }
        .search-section { padding: 40px; background: #f8f9fa; }
        .search-box { display: flex; gap: 12px; margin-bottom: 24px; }
        .search-input {
            flex: 1;
            padding: 16px 20px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 12px;
        }
        .search-input:focus { border-color: #667eea; outline: none; }
        .search-btn {
            padding: 16px 44px;
            font-size: 16px;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 12px;
            cursor: pointer;
        }
        .search-btn:hover { transform: translateY(-2px); }
        .search-btn:disabled { opacity: 0.6; cursor: not-allowed; }
        .results-section { padding: 40px; }
        .loading {
            text-align: center;
            padding: 80px 20px;
            font-size: 1.3em;
            color: #667eea;
        }
        .ai-response {
            background: linear-gradient(135deg, #f0f8ff, #f3e8ff);
            border: 2px solid #667eea;
            border-radius: 16px;
            padding: 28px;
            margin-bottom: 40px;
        }
        .result-category { margin: 0 0 48px; }
        .category-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 3px solid #667eea;
        }
        .category-title { font-size: 1.6em; color: #222; }
        .result-card {
            background: #fff;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
        }
        .image-container img {
            max-width: 100%;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .no-results { text-align: center; padding: 100px 20px; color: #777; }
        .examples {
            background: #fff9e6;
            padding: 20px;
            border-radius: 12px;
            margin-top: 20px;
        }
        .examples h4 { color: #d35400; margin-bottom: 12px; }
        .examples ul { list-style: none; padding-left: 0; margin: 0; }
        .examples li {
            padding: 8px 0;
            cursor: pointer;
            color: #444;
        }
        .examples li:hover { color: #667eea; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>NCERT Physics Class 11 – AI RAG</h1>
            <div class="ai-badge">Powered by LLM</div>
            <p>Ask anything from your textbook</p>
        </div>

        <div class="search-section">
            <div class="search-box">
                <input type="text" id="queryInput" class="search-input" 
                       placeholder="Ask anything... (e.g. Explain stress and strain with diagram)">
                <button class="search-btn" id="searchBtn">Search</button>
            </div>

            <div class="examples">
                <h4>Quick examples:</h4>
                <ul id="examplesList">
                    <li>Explain stress and strain with diagrams</li>
                    <li>What is Young's modulus? Show formula</li>
                    <li>Give me problems on elasticity</li>
                    <li>Which chapters discuss mechanical properties?</li>
                    <li>Summarize Chapter 8</li>
                </ul>
            </div>
        </div>

        <div class="results-section" id="resultsSection">
            <div style="text-align:center; padding:100px 20px; color:#777;">
                <p style="font-size:1.4em;">Type your question above ↑</p>
            </div>
        </div>
    </div>

    <script>
    console.log('🚀 Script loaded successfully');
    
    document.addEventListener('DOMContentLoaded', () => {
        console.log('✅ DOM loaded');
        
        const input = document.getElementById('queryInput');
        const btn   = document.getElementById('searchBtn');
        const res   = document.getElementById('resultsSection');
        const exList = document.getElementById('examplesList');

        if (!input || !btn || !res || !exList) {
            console.error('❌ Missing DOM elements:', {input, btn, res, exList});
            return;
        }
        console.log('✅ All DOM elements found');

        // Enter key support
        input.addEventListener('keypress', e => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                console.log('⏎ Enter key pressed');
                performSearch();
            }
        });

        // Search button
        btn.addEventListener('click', () => {
            console.log('🔍 Search button clicked');
            performSearch();
        });

        // Click on example items (delegation - no inline onclick)
        exList.addEventListener('click', e => {
            const li = e.target.closest('li');
            if (li) {
                console.log('📝 Example clicked:', li.textContent);
                input.value = li.textContent.trim();
                performSearch();
            }
        });

        async function performSearch() {
            const q = input.value.trim();
            console.log('🔍 performSearch called with query:', q);
            
            if (!q) {
                console.warn('⚠️ Empty query');
                alert('Please type a question');
                return;
            }

            res.innerHTML = '<div class="loading">Searching textbook & generating answer...</div>';
            btn.disabled = true;
            btn.textContent = 'Processing...';

            try {
                console.log('📡 Sending API request...');
                const r = await fetch('/api/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        query: q,
                        max_results: 10
                    })
                });

                console.log('📥 Response status:', r.status);

                if (!r.ok) {
                    const errText = await r.text().catch(() => '');
                    console.error('❌ API error:', r.status, errText);
                    throw new Error(`Server error ${r.status}${errText ? ': ' + errText : ''}`);
                }

                const data = await r.json();
                console.log('✅ Data received:', data);
                showResults(data, q);
            } catch (err) {
                console.error('❌ Search error:', err);
                res.innerHTML = `
                    <div class="no-results">
                        <p style="color:red; font-size:1.3em;">Error</p>
                        <p>${err.message}</p>
                    </div>`;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Search';
            }
        }

        function showResults(data, q) {
            console.log('🎨 Rendering results...');
            
            let html = `
                <div style="padding:20px; background:#f0f8ff; border-radius:12px; margin-bottom:30px;">
                    <h3 style="margin:0 0 8px;">Question: ${escapeHtml(q)}</h3>
                </div>`;

            if (data.llm_response) {
                console.log('🤖 LLM response found');
                html += `
                    <div class="ai-response">
                        <div style="font-size:1.3em; color:#444; margin-bottom:16px;">🤖 AI Answer</div>
                        <div style="line-height:1.7;">${formatMarkdown(data.llm_response)}</div>
                    </div>`;
            }

            const r = data.results || {};
            let has = false;

            if (r.text?.length)     { has = true; html += block('Text', r.text, fmtText); }
            if (r.equations?.length){ has = true; html += block('Equations', r.equations, fmtEq); }
            if (r.problems?.length) { has = true; html += block('Problems', r.problems, fmtProb); }
            if (r.tables?.length)   { has = true; html += block('Tables', r.tables, fmtTable); }
            if (r.images?.length)   { has = true; html += block('Diagrams', r.images, fmtImg); }

            if (!has && !data.llm_response) {
                console.log('⚠️ No results found');
                html += '<div class="no-results"><p>No results found</p></div>';
            }

            res.innerHTML = html;
            console.log('✅ Results rendered');
        }

        function block(title, items, fn) {
            let s = `<div class="result-category"><h2>${title}</h2>`;
            items.forEach(i => s += fn(i));
            s += '</div>';
            return s;
        }

        function fmtText(i) {
            const m = i.metadata || {};
            return `<div class="result-card">
                <div style="font-weight:bold;">Chapter ${m.chapter_number || '?'} - Page ${m.page || '?'}</div>
                <div>${escapeHtml(i.content || '')}</div>
            </div>`;
        }

        function fmtEq(i) {
            const m = i.metadata || {};
            return `<div class="result-card">
                <div style="font-weight:bold;">Eq. from Ch ${m.chapter_number || '?'} - p${m.page || '?'}</div>
                <pre>${escapeHtml(i.content || '')}</pre>
            </div>`;
        }

        function fmtProb(i) {
            const m = i.metadata || {};
            return `<div class="result-card">
                <div style="font-weight:bold;">Problem from Ch ${m.chapter_number || '?'} - p${m.page || '?'}</div>
                <div>${escapeHtml(i.content || '').replace(/\n/g,'<br>')}</div>
            </div>`;
        }

        function fmtTable(i) {
            const m = i.metadata || {};
            const td = i.table_data || {};
            return `<div class="result-card">
                <div style="font-weight:bold;">Table from Ch ${m.chapter_number || '?'} - p${m.page || '?'}</div>
                ${td.markdown ? mdTable(td.markdown) : ''}
            </div>`;
        }

        function fmtImg(i) {
            const m = i.metadata || {};
            const d = i.image_data || {};
            return `<div class="result-card">
                <div style="font-weight:bold;">Figure from Ch ${m.chapter_number || '?'} - p${m.page || '?'}</div>
                <div class="image-container">
                    <img src="/${d.path || ''}" alt="Diagram">
                    ${d.ocr_text ? `<p>${escapeHtml(d.ocr_text)}</p>` : ''}
                </div>
            </div>`;
        }

        function mdTable(md) {
            const lines = md.trim().split('\n');
            if (lines.length < 2) return md;
            let h = '<table style="width:100%;border-collapse:collapse;">';
            const heads = lines[0].split('|').map(x=>x.trim()).filter(Boolean);
            h += '<tr>' + heads.map(x=>`<th style="padding:10px;border:1px solid #ddd;background:#667eea;color:white;">${x}</th>`).join('') + '</tr>';
            for (let i = 2; i < lines.length; i++) {
                const cells = lines[i].split('|').map(x=>x.trim()).filter(Boolean);
                if (cells.length) h += '<tr>' + cells.map(x=>`<td style="padding:10px;border:1px solid #ddd;">${x}</td>`).join('') + '</tr>';
            }
            h += '</table>';
            return h;
        }

        function escapeHtml(t) {
            console.log('🔒 Escaping HTML');
            const d = document.createElement('div');
            d.textContent = t;
            return d.innerHTML;
        }

        function formatMarkdown(t) {
            console.log('📝 Formatting markdown');
            try {
                let h = escapeHtml(t);
                // Bold text with **text**
                h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
                // Newlines to breaks
                h = h.replace(/\n/g, '<br>');
                return h;
            } catch (err) {
                console.error('❌ Markdown formatting error:', err);
                return escapeHtml(t);
            }
        }
        
        console.log('✅ All event listeners attached');
    });
    </script>
</body>
</html>
    """


# ────────────────────────────────────────────────
# API Routes
# ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home():
    print("📄 Serving home page")
    return get_html_template()


@app.post("/api/query")
async def query(request: QueryRequest):
    print(f"🔍 Query received: {request.query}")
    try:
        results = engine.execute_query(
            request.query,
            max_results=request.max_results
        )
        print(f"✅ Query executed successfully")
        return results
    except Exception as e:
        import traceback
        print(f"❌ Error processing query:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)