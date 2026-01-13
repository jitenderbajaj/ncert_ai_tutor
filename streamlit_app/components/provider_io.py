"""
Provider I/O component for Streamlit
Displays LLM prompts and outputs with syntax highlighting
Version: 0.12.5 - Universal compatibility with robust None handling
"""

import streamlit as st
from typing import List, Dict, Any


def render_provider_io_entry(entry: Dict[str, Any], index: int, expanded: bool = False):
    """
    Render a single provider I/O entry with robust error handling
    
    Args:
        entry: I/O entry dict
        index: Entry index for unique keys
        expanded: Whether to expand by default
    """
    try:
        # Validate entry is a dictionary
        if not isinstance(entry, dict):
            st.error(f"[{index}] Invalid entry (not a dictionary): {type(entry)}")
            return
        
        # Extract fields with safe defaults
        timestamp = entry.get("timestamp", "Unknown")
        provider = entry.get("provider", "Unknown")
        model = entry.get("model", "Unknown")
        duration_ms = entry.get("duration_ms", 0)
        error = entry.get("error")
        
        # Status icon
        status_icon = "✅" if not error else "❌"
        
        # Expander header (unique per entry)
        header = f"{status_icon} [{index}] {provider}/{model} — {duration_ms}ms"
        
        with st.expander(header, expanded=expanded):
            # Metrics row (NO KEYS - not supported in old Streamlit)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(f"Provider [{index}]", provider)
            with col2:
                st.metric(f"Model [{index}]", model)
            with col3:
                st.metric(f"Duration [{index}]", f"{duration_ms}ms")
            with col4:
                prompt_length = entry.get("prompt_length", 0)
                st.metric(f"Prompt Size [{index}]", f"{prompt_length} chars")
            
            # Correlation ID with None check
            correlation_id = entry.get("correlation_id") or "N/A"
            st.markdown(f"**Correlation ID:** `{correlation_id}`")
            
            # Timestamp
            st.caption(f"📅 {timestamp}")
            
            # Error display
            if error:
                st.error(f"**Error:** {error}")
            
            st.divider()
            
            # Prompt section
            st.subheader("📝 Prompt (Input to LLM)")
            prompt = entry.get("prompt_sanitized") or entry.get("prompt") or ""

            if prompt:
                with st.container():
                    # Display as-is (no line numbers added)
                    st.text_area(
                        label=f"Prompt Content [{index}]",
                        value=prompt,
                        height=min(300, prompt.count('\n') * 22 + 100),
                        disabled=True,
                        label_visibility="collapsed"
                    )
                
                # Download button
                safe_corr_id = correlation_id[:8] if correlation_id and correlation_id != "N/A" else "unknown"
                st.download_button(
                    label=f"💾 Download Prompt [{index}]",
                    data=prompt,
                    file_name=f"prompt_{safe_corr_id}.txt",
                    mime="text/plain"
                )
            else:
                st.warning("No prompt captured")

            st.divider()

            # Output section
            st.subheader("💬 Output (Response from LLM)")
            output = entry.get("output") or ""
            output_length = entry.get("output_length", 0)

            if output:
                # Token count estimate
                token_estimate = output_length // 4 if output_length else 0
                st.info(f"📊 Length: {output_length} chars (~{token_estimate} tokens)")
                
                with st.container():
                    # Display as-is (no line numbers added)
                    st.text_area(
                        label=f"Output Content [{index}]",
                        value=output,
                        height=min(400, output.count('\n') * 22 + 100),
                        disabled=True,
                        label_visibility="collapsed"
                    )
                
                # Download button
                safe_corr_id = correlation_id[:8] if correlation_id and correlation_id != "N/A" else "unknown"
                st.download_button(
                    label=f"💾 Download Output [{index}]",
                    data=output,
                    file_name=f"output_{safe_corr_id}.txt",
                    mime="text/plain"
                )
            else:
                st.warning("No output (request may have failed)")

            
    except Exception as e:
        st.error(f"[{index}] Error rendering entry: {e}")
        # Debug expander with unique header
        with st.expander(f"🐛 Debug Info [{index}]"):
            st.write(f"Entry type: {type(entry)}")
            st.write(f"Entry content: {entry}")


def render_provider_io_panel(entries: List[Dict[str, Any]]):
    """
    Render provider I/O panel with multiple entries
    
    Args:
        entries: List of I/O entry dicts
    """
    if not entries:
        st.info("No provider I/O entries yet. Make queries in the Tutor tab to see LLM activity.")
        return
    
    # Filter out invalid entries
    valid_entries = []
    invalid_count = 0
    
    for entry in entries:
        if isinstance(entry, dict):
            valid_entries.append(entry)
        else:
            invalid_count += 1
    
    if invalid_count > 0:
        st.warning(f"⚠️ Skipped {invalid_count} invalid entries")
    
    st.success(f"📊 Showing {len(valid_entries)} recent I/O entries")
    
    # Render each entry (most recent first)
    for i, entry in enumerate(reversed(valid_entries)):
        try:
            render_provider_io_entry(entry, index=i+1, expanded=(i == 0))
        except Exception as e:
            st.error(f"Error rendering entry {i+1}: {e}")




