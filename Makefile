# ============================================================
# Patch-based install/update/uninstall — Import/Export Jobs
#
# Usage:
#   make install   DEST=/path/to/other/ai-toolkit
#   make update    DEST=/path/to/other/ai-toolkit
#   make uninstall DEST=/path/to/other/ai-toolkit
#
# I file modificati vengono applicati via patch per evitare di
# sovrascrivere versioni upstream più recenti nella destinazione.
# ============================================================

.DEFAULT_GOAL := help

.PHONY: help install update uninstall zip _check-dest _deploy-new-files

# ── Help ─────────────────────────────────────────────────
help:
	@echo ""
	@echo "Utilizzo: make <target> DEST=/path/to/ai-toolkit"
	@echo ""
	@echo "  install     Installa la funzionalità export/import job"
	@echo "  update      Aggiorna la funzionalità export/import job"
	@echo "  uninstall   Rimuove la funzionalità export/import job"
	@echo "  zip         Crea importexport-dist.zip con i file necessari all'installazione"
	@echo ""

OUT ?= .
ZIP_FILE := $(OUT)/importexport-dist.zip
ZIP_FILES := Makefile \
	patches/BaseSDTrainProcess.patch \
	patches/next_config.patch \
	patches/jobs_page.patch \
	patches/JobActionBar.patch \
	ui/src/app/api/jobs/[jobID]/export/route.ts \
	ui/src/app/api/jobs/import/route.ts \
	ui/src/components/ImportJobModal.tsx

# ── Zip ───────────────────────────────────────────────────
zip:
	@if command -v zip > /dev/null 2>&1; then \
		zip -r "$(ZIP_FILE)" $(ZIP_FILES); \
	else \
		python3 -c "import zipfile,sys; files=sys.argv[1:]; z=zipfile.ZipFile(files[0],'w',zipfile.ZIP_DEFLATED); [z.write(f) for f in files[1:]]; z.close()" \
			"$(ZIP_FILE)" $(ZIP_FILES); \
	fi
	@echo "✓ $(ZIP_FILE) creato"

# ── Validazione percorso ──────────────────────────────────
_check-dest:
	@test -n "$(DEST)" || { echo "Errore: specifica il percorso, es:  make install DEST=/path/to/ai-toolkit"; exit 1; }
	@test -d "$(DEST)/ui" || { echo "Errore: '$(DEST)' non sembra una directory ai-toolkit valida (manca ui/)"; exit 1; }
	@test -d "$(DEST)/jobs/process" || { echo "Errore: '$(DEST)' non sembra una directory ai-toolkit valida (manca jobs/process/)"; exit 1; }

# ── Copia file completamente nuovi + npm ──────────────────
_deploy-new-files:
	@echo ""
	@echo "==> Copia file nuovi / aggiornati..."
	@mkdir -p "$(DEST)/ui/src/app/api/jobs/[jobID]/export"
	@cp "ui/src/app/api/jobs/[jobID]/export/route.ts" \
		"$(DEST)/ui/src/app/api/jobs/[jobID]/export/route.ts"
	@echo "    ui/src/app/api/jobs/[jobID]/export/route.ts"

	@mkdir -p "$(DEST)/ui/src/app/api/jobs/import"
	@cp "ui/src/app/api/jobs/import/route.ts" \
		"$(DEST)/ui/src/app/api/jobs/import/route.ts"
	@echo "    ui/src/app/api/jobs/import/route.ts"

	@cp "ui/src/components/ImportJobModal.tsx" \
		"$(DEST)/ui/src/components/ImportJobModal.tsx"
	@echo "    ui/src/components/ImportJobModal.tsx"

	@echo ""
	@echo "==> Sincronizza pacchetti npm..."
	@cd "$(DEST)/ui" && npm install unzip-stream busboy @types/busboy

# ── Install ───────────────────────────────────────────────
install: _check-dest
	@echo ""
	@echo "==> Backup file che verranno modificati (salvati come .aitk-bak)..."
	@for f in \
		"jobs/process/BaseSDTrainProcess.py" \
		"ui/next.config.ts" \
		"ui/src/app/jobs/page.tsx" \
		"ui/src/components/JobActionBar.tsx"; do \
		if [ -f "$(DEST)/$$f" ] && [ ! -f "$(DEST)/$$f.aitk-bak" ]; then \
			cp "$(DEST)/$$f" "$(DEST)/$$f.aitk-bak"; \
			echo "    backup: $$f"; \
		elif [ -f "$(DEST)/$$f.aitk-bak" ]; then \
			echo "    skip:   $$f (backup già presente)"; \
		else \
			echo "    WARN:   $$f non trovato nella destinazione"; \
		fi; \
	done
	@$(MAKE) -s _deploy-new-files DEST="$(DEST)"
	@echo ""
	@echo "==> Applica patch ai file modificati..."
	@apply_patch() { \
		pf="$$1"; label="$$2"; \
		if patch -p1 -d "$(DEST)" --dry-run < "$$pf" > /dev/null 2>&1; then \
			patch -p1 -d "$(DEST)" --forward --quiet < "$$pf"; \
			echo "    patched: $$label"; \
		elif patch -p1 -d "$(DEST)" --dry-run --reverse < "$$pf" > /dev/null 2>&1; then \
			echo "    skip:    $$label (patch già applicata)"; \
		else \
			echo ""; \
			echo "    CONFLITTO: $$label"; \
			echo "    Risolvi manualmente: patch -p1 -d \"$(DEST)\" < $$pf"; \
			return 1; \
		fi; \
	}; \
	apply_patch patches/BaseSDTrainProcess.patch "jobs/process/BaseSDTrainProcess.py" || exit 1; \
	apply_patch patches/next_config.patch        "ui/next.config.ts"                  || exit 1; \
	apply_patch patches/jobs_page.patch          "ui/src/app/jobs/page.tsx"            || exit 1; \
	apply_patch patches/JobActionBar.patch       "ui/src/components/JobActionBar.tsx"  || exit 1
	@echo ""
	@echo "✓ Import/Export installato. Riavvia il server ai-toolkit."
	@echo ""

# ── Update ────────────────────────────────────────────────
update: _check-dest
	@echo ""
	@echo "==> Aggiornamento file nuovi..."
	@$(MAKE) -s _deploy-new-files DEST="$(DEST)"
	@echo ""
	@echo "==> Aggiornamento file modificati (reverse + ri-applica patch)..."
	@update_patched() { \
		pf="$$1"; rel="$$2"; \
		if patch -p1 -d "$(DEST)" --dry-run --reverse --quiet < "$$pf" > /dev/null 2>&1; then \
			patch -p1 -d "$(DEST)" --reverse --quiet < "$$pf"; \
		fi; \
		if patch -p1 -d "$(DEST)" --dry-run --quiet < "$$pf" > /dev/null 2>&1; then \
			patch -p1 -d "$(DEST)" --forward --quiet < "$$pf"; \
			echo "    updated: $$rel"; \
		elif patch -p1 -d "$(DEST)" --dry-run --reverse --quiet < "$$pf" > /dev/null 2>&1; then \
			echo "    skip:    $$rel (già aggiornato)"; \
		else \
			echo ""; \
			echo "    CONFLITTO: $$rel"; \
			echo "    Risolvi manualmente: patch -p1 -d \"$(DEST)\" < $$pf"; \
			return 1; \
		fi; \
	}; \
	update_patched patches/BaseSDTrainProcess.patch "jobs/process/BaseSDTrainProcess.py" || exit 1; \
	update_patched patches/next_config.patch        "ui/next.config.ts"                  || exit 1; \
	update_patched patches/jobs_page.patch          "ui/src/app/jobs/page.tsx"            || exit 1; \
	update_patched patches/JobActionBar.patch       "ui/src/components/JobActionBar.tsx"  || exit 1
	@echo ""
	@echo "✓ Import/Export aggiornato. Riavvia il server ai-toolkit."
	@echo ""

# ── Uninstall ─────────────────────────────────────────────
uninstall: _check-dest
	@echo ""
	@echo "==> Rimuovi file aggiunti..."
	@rm -f "$(DEST)/ui/src/app/api/jobs/[jobID]/export/route.ts"
	@echo "    removed: ui/src/app/api/jobs/[jobID]/export/route.ts"
	@rmdir "$(DEST)/ui/src/app/api/jobs/[jobID]/export" 2>/dev/null || true

	@rm -f "$(DEST)/ui/src/app/api/jobs/import/route.ts"
	@echo "    removed: ui/src/app/api/jobs/import/route.ts"
	@rmdir "$(DEST)/ui/src/app/api/jobs/import" 2>/dev/null || true

	@rm -f "$(DEST)/ui/src/components/ImportJobModal.tsx"
	@echo "    removed: ui/src/components/ImportJobModal.tsx"

	@echo ""
	@echo "==> Rimuovi patch dai file modificati..."
	@unpatch() { \
		pf="$$1"; rel="$$2"; dest_file="$(DEST)/$$rel"; \
		if patch -p1 -d "$(DEST)" --dry-run --reverse < "$$pf" > /dev/null 2>&1; then \
			patch -p1 -d "$(DEST)" --reverse --quiet < "$$pf"; \
			rm -f "$$dest_file.aitk-bak"; \
			echo "    unpatched: $$rel"; \
		elif [ -f "$$dest_file.aitk-bak" ]; then \
			mv "$$dest_file.aitk-bak" "$$dest_file"; \
			echo "    restored:  $$rel (patch non applicabile, ripristinato dal backup)"; \
		else \
			echo "    WARN:      $$rel — patch non applicabile e nessun backup trovato (skip)"; \
		fi; \
	}; \
	unpatch patches/BaseSDTrainProcess.patch "jobs/process/BaseSDTrainProcess.py"; \
	unpatch patches/next_config.patch        "ui/next.config.ts"; \
	unpatch patches/jobs_page.patch          "ui/src/app/jobs/page.tsx"; \
	unpatch patches/JobActionBar.patch       "ui/src/components/JobActionBar.tsx"

	@echo ""
	@echo "==> Rimuovi pacchetti npm..."
	@cd "$(DEST)/ui" && npm uninstall unzip-stream busboy @types/busboy

	@echo ""
	@echo "✓ Import/Export disinstallato. Riavvia il server ai-toolkit."
	@echo ""
