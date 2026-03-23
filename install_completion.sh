#!/usr/bin/env bash
# install_completion.sh — installe la complétion bash pour `uv run agent`
set -euo pipefail

COMPLETION_FILE="$HOME/.agent_system_completion.bash"
BASHRC="$HOME/.bashrc"
SOURCE_LINE="source \"$COMPLETION_FILE\""

echo "==> Génération du script de complétion personnalisé..."

cat > "$COMPLETION_FILE" << 'EOF'
# Complétion bash pour `uv run agent`
# Appelle directement le mécanisme interne de Typer via _AGENT_COMPLETE.
_uv_run_agent_completion() {
    # N'intervenir que si la ligne contient `uv run agent`
    if [[ "$COMP_LINE" =~ ^uv[[:space:]]+run[[:space:]]+agent ]]; then
        # Reconstruit la ligne et le curseur comme si `agent` était la commande racine
        local stripped_line="${COMP_LINE#*agent}"
        local agent_words=( "agent" $stripped_line )
        local agent_cword=$(( ${#agent_words[@]} - 1 ))

        COMPREPLY=( $(
            env COMP_WORDS="${agent_words[*]}" \
                COMP_CWORD="$agent_cword" \
                _AGENT_COMPLETE=complete_bash \
                uv run agent
        ) )
    fi
}
complete -o default -F _uv_run_agent_completion uv
EOF

echo "    ✓ Script de complétion écrit dans $COMPLETION_FILE"

# Ajoute le source dans .bashrc seulement s'il n'y est pas déjà
if grep -qF "$SOURCE_LINE" "$BASHRC" 2>/dev/null; then
    echo "    ✓ Source déjà présent dans $BASHRC — rien à faire"
else
    printf '\n# Agent System — complétion bash\n%s\n' "$SOURCE_LINE" >> "$BASHRC"
    echo "    ✓ Source ajouté dans $BASHRC"
fi

echo ""
echo "==> Installation terminée. Rechargez votre shell :"
echo ""
echo "        source ~/.bashrc"
echo ""
echo "    Puis testez :"
echo ""
echo "        uv run agent <TAB>"
echo "        uv run agent core <TAB>"