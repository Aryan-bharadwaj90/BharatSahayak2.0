import base64
import json
import re
from html import escape
from pathlib import Path

import gradio as gr

from src.data_pipeline import process_documents, read_schemes
from src.eligibility import check_eligibility
from src.rag import generate_answer
from src.retrieval import build_index_from_file, load_index, retrieve


ROOT = Path(__file__).resolve().parent
CATALOG = read_schemes()
KB_READY = False
EXAMPLES = [
    ("PM_KISAN", "PM-KISAN", "Income support for farmers", "What benefits does PM-KISAN offer?",
     "पीएम किसान योजना के क्या लाभ हैं?", "farm"),
    ("AYUSHMAN_BHARAT_PMJAY", "Ayushman Bharat", "Healthcare for families", "Who can qualify for Ayushman Bharat PM-JAY?",
     "आयुष्मान भारत पीएम-जय के लिए कौन पात्र है?", "health"),
    ("PMAY_URBAN", "PMAY-Urban", "Housing in urban areas", "What documents do I need for PMAY-Urban?",
     "पीएमएवाई शहरी के लिए कौन से दस्तावेज़ चाहिए?", "home"),
]


def initialize():
    global KB_READY
    if not KB_READY:
        try:
            try:
                load_index()
            except (OSError, ValueError, RuntimeError):
                if not process_documents():
                    return "The scheme library is empty."
                build_index_from_file()
            KB_READY = True
        except Exception as exc:
            return f"Could not open the scheme library. {exc}"
    return "Scheme library ready"


def evidence_excerpt(chunk):
    section = chunk.get("section_type", "")
    text = chunk.get("text", "")
    if section != "official_pdf":
        marker = f"Information Section: {section} "
        _, found, text = text.partition(marker)
        if not found:
            return ""
    text = re.split(r"(?:^|\s)logic:\s", text, maxsplit=1)[0].strip()
    return text


def answer_user(query, profile_text, language, selected_scheme):
    if not query or not query.strip():
        return "Enter a question or choose one of the suggested schemes."
    if not KB_READY:
        status = initialize()
        if not KB_READY:
            return status
    profile = None
    if profile_text and profile_text.strip():
        try:
            profile = json.loads(profile_text)
        except json.JSONDecodeError:
            return 'Check the profile JSON. For example, {"residence_area": "urban"}.'
        if not isinstance(profile, dict):
            return "Profile must be a JSON object."

    try:
        chunks = retrieve(query, top_k=5, scheme_id=selected_scheme)
        if not chunks:
            return "No matching evidence found. Try a different question or scheme."
        eligibility = None
        if profile is not None and selected_scheme:
            eligibility = check_eligibility(selected_scheme, profile)
        result = generate_answer(query, chunks, eligibility, language)
        paragraphs = [result.get("answer", "No answer generated.")]
        if result.get("error"):
            hindi = language == "hi"
            paragraphs = [
                "### स्रोत से अंश" if hindi else "### From the scheme library",
                ("AI उत्तर अभी उपलब्ध नहीं है। नीचे मूल स्रोत के अंग्रेज़ी अंश हैं।"
                 if hindi else "AI generation is unavailable. Here are excerpts from the retrieved sources."),
            ]
            for chunk in chunks:
                excerpt = evidence_excerpt(chunk)
                if excerpt:
                    label = chunk.get("section_type", "").replace("_", " ").capitalize()
                    paragraphs.append(f"**{chunk['scheme_name']} · {label}**\n\n{excerpt}")
                if len(paragraphs) == 5:
                    break
            if len(paragraphs) == 2:
                paragraphs.append(result["answer"])

        if eligibility:
            statuses = {"satisfied": "Listed conditions met", "not_satisfied": "A condition was not met",
                        "unknown": "More information needed"}
            paragraphs.extend(["### Preliminary eligibility", f"**{statuses[eligibility['status']]}**"])
            for key, label in (("satisfied", "Met"), ("not_satisfied", "Unmet"), ("unknown", "To verify")):
                details = eligibility.get(key) or []
                if details:
                    paragraphs.append(f"**{label}**\n\n" + "\n".join(f"- {item}" for item in details))
        elif profile is not None:
            paragraphs.append("Select a scheme to check preliminary eligibility for this profile.")

        sources = result.get("sources", [])
        if sources:
            paragraphs.append("### Official sources")
            paragraphs.append("\n".join(
                f"- [{s['source_title']}]({s['source_url']})" if s.get("source_url") else f"- {s['source_title']}"
                for s in sources
            ))
        return "\n\n".join(paragraphs)
    except Exception as exc:
        return f"Could not complete this search. {exc}"


def choose_example(scheme, english, hindi):
    def fill(language):
        return (hindi if language == "hi" else english), scheme
    return fill


art = base64.b64encode((ROOT / "assets/village.webp").read_bytes()).decode("ascii")
directory_links = "".join(
    f'<a href="{escape(item["official_url"], quote=True)}" target="_blank" rel="noopener noreferrer">'
    f'{escape(item["scheme_name"])} <span aria-hidden="true">↗</span></a>'
    for item in CATALOG
)

with gr.Blocks(title="BharatSahayak") as demo:
    gr.HTML("""
    <header class="mast"><a class="brand" href="#welcome"><span class="brandmark" aria-hidden="true">✺</span>
    Bharat<span class="brand-end">Sahayak</span></a>
    <nav aria-label="Main navigation"><a href="#ask">Ask a question</a><a href="#directory">Scheme directory</a></nav></header>
    """)
    gr.HTML(f"""
    <section class="hero" id="welcome">
      <div class="hero-copy"><div class="kicker"><i></i> SCHEME ASSISTANT</div>
      <h1>Government schemes,<br><em>made simple.</em></h1>
      <p>Explore benefits, check eligibility, and find the right documents for your application.</p>
      <div class="hero-meta"><span>{len(CATALOG)} schemes</span><b>·</b><span>English &amp; हिन्दी</span></div></div>
      <img alt="Illustrated Indian village with terraced fields and a rising sun" src="data:image/webp;base64,{art}">
    </section>
    """)
    with gr.Row(elem_id="workspace"):
        with gr.Column(scale=7, min_width=320, elem_classes="panel", elem_id="ask"):
            gr.HTML('<div class="kicker"><i></i> ASK THE ASSISTANT</div><h2 class="section-title">How can we help?</h2>')
            query = gr.Textbox(label="Your question", placeholder="Which schemes can help a farmer with crop insurance?", lines=3)
            with gr.Row(elem_classes="options"):
                language = gr.Dropdown([("English", "en"), ("हिन्दी", "hi")], value="en", label="Answer language", interactive=True)
                selected_scheme = gr.Dropdown([("All schemes", "")] + [(s["scheme_name"], s["scheme_id"]) for s in CATALOG],
                                              value="", label="Search within", interactive=True)
            with gr.Accordion("Add an eligibility profile (optional)", open=False):
                profile = gr.Textbox(label="Profile (JSON)", lines=3, placeholder='{"residence_area": "urban"}',
                                     info="Choose a scheme above. Four schemes have preliminary eligibility rules.")
            with gr.Row(elem_classes="action-row"):
                submit = gr.Button("Find guidance", variant="primary", elem_id="submit-guidance")
                status = gr.Markdown("Opening scheme library…", elem_id="kb-status")
        with gr.Column(scale=4, min_width=275, elem_classes="panel", elem_id="examples"):
            gr.HTML('<div class="kicker"><i></i> QUICK START</div><h2 class="section-title">Try a scheme</h2><p class="hint">Choose a scheme to fill in a sample question.</p>')
            for sid, title, subtitle, english, hindi, style in EXAMPLES:
                suggestion = gr.Button(f"{title}\n{subtitle}", elem_classes=["scheme-card", style], size="lg")
                suggestion.click(choose_example(sid, english, hindi), language, [query, selected_scheme], queue=False)
            gr.HTML('<a class="directory-link" href="#directory">Browse all scheme portals <span aria-hidden="true">↓</span></a>')
    output = gr.Markdown("", elem_id="answer", label="Scheme guidance", sanitize_html=True)
    submit.click(answer_user, [query, profile, language, selected_scheme], output, api_name="answer_user",
                 show_progress="full", scroll_to_output=True)
    demo.load(initialize, outputs=status, show_progress="hidden", api_name="initialize")
    gr.HTML(f"""
    <section id="directory"><div class="kicker"><i></i> OFFICIAL SOURCES</div>
    <h2 class="section-title">Scheme directory</h2><div class="directory-grid">{directory_links}</div></section>
    <footer class="site-footer"><strong>BharatSahayak</strong><span>Preliminary guidance. Confirm eligibility with the relevant authority.</span></footer>
    """)


def launch(**options):
    return demo.launch(css_paths=[ROOT / "assets/theme.css"], footer_links=[], **options)


if __name__ == "__main__":
    launch(share=False)
