.PHONY: help pdf clean validate

help:
	@echo "🎓 PITT Academic Research Publications"
	@echo ""
	@echo "Commands:"
	@echo "  make pdf              - Render ALL publications to PDF"
	@echo "  make pdf [name]       - Render specific publication (e.g., make pdf spotify_popularity)"
	@echo "  make clean            - Remove generated PDFs and temp files"
	@echo "  make validate         - Check publication formatting"
	@echo ""
	@echo "Available Publications:"
	@echo "  - spotify_popularity     Predicting Song Popularity with ML"
	@echo "  - fire_safety_dashboard  Fire Safety Data Analytics"
	@echo "  - manufacturing_analytics Manufacturing Process Analysis"
	@echo "  - network_analysis       College Football Network Centrality"
	@echo ""
	@echo "Requirements:"
	@echo "  - Quarto: brew install --cask quarto"
	@echo "  - Python: 3.8+ with venv activated"
	@echo "  - LaTeX: Included with Quarto"
	@echo ""
	@echo "Example:"
	@echo "  make pdf spotify_popularity  # Render single publication"
	@echo "  make pdf                     # Render all publications"

pdf:
	@echo "📚 Generating academic publications..."
	@bash ../quarto/scripts/generate_pdfs.sh $(filter-out $@,$(MAKECMDGOALS))
	@echo ""
	@echo "✅ PDFs available in: publications/pdf/"

clean:
	@echo "🧹 Cleaning generated files..."
	@rm -rf publications/pdf/*.pdf
	@rm -rf publications/*/.quarto/
	@rm -rf publications/*/pdf/
	@rm -rf publications/*/*.log
	@rm -rf publications/*/*.aux
	@rm -rf publications/*/*.tex
	@find publications/ -name "*_files" -type d -exec rm -rf {} + 2>/dev/null || true
	@find publications/ -name "*_cache" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete"

validate:
	@echo "🔍 Validating publication formatting..."
	@echo "Checking for common issues:"
	@echo ""
	@echo "1. Blank lines after bold headers:"
	@grep -rn "^\*\*.*:\*\*$$" publications/*/index.qmd || echo "   ✅ No issues found"
	@echo ""
	@echo "2. Indented code blocks:"
	@grep -rn "^[[:space:]]\+\`\`\`" publications/*/index.qmd && echo "   ❌ Found indented backticks" || echo "   ✅ No indented backticks"
	@echo ""
	@echo "3. Mermaid diagrams (should use Plotly):"
	@grep -rn "mermaid" publications/*/index.qmd && echo "   ⚠️  Found mermaid (use Plotly)" || echo "   ✅ No mermaid diagrams"
	@echo ""
	@echo "For complete guidelines, see: ../quarto/RESEARCH_GUIDELINES.md"

# Allow passing publication name as argument
%:
	@:
