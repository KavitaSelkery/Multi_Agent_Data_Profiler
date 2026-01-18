"""
UI Components for Streamlit Dashboard
"""
import streamlit as st
from typing import Optional

def render_header(title: str):
    """Render the main header"""
    st.markdown(f"""
    <h1 class="main-header">{title}</h1>
    """, unsafe_allow_html=True)

def render_sidebar(dashboard):
    """Render the sidebar with connection form and controls"""
    with st.sidebar:
        st.title("🔗 Connection")
        
        if not st.session_state.snowflake_connected:
            # Connection form
            with st.form("snowflake_connection"):
                st.subheader("Connect to Snowflake")
                
                # Correct parameter names for Snowflake
                account = st.text_input("Account", placeholder="your-account", 
                                       help="e.g., xy12345.us-east-1")
                user = st.text_input("User", placeholder="your-username", 
                                    help="Snowflake username (not email)")
                password = st.text_input("Password", type="password", 
                                        placeholder="your-password")
                warehouse = st.text_input("Warehouse", placeholder="COMPUTE_WH",
                                         help="Optional")
                database = st.text_input("Database", placeholder="your-database",
                                        help="Optional")
                schema = st.text_input("Schema", placeholder="PUBLIC",
                                      help="Optional")
                role = st.text_input("Role", placeholder="your-role",
                                    help="Optional")
                
                openai_key = st.text_input("OpenAI API Key", type="password", 
                                          placeholder="sk-...", 
                                          help="Required for AI agents")
                
                col1, col2 = st.columns(2)
                with col1:
                    connect_button = st.form_submit_button("Connect", type="primary")
                with col2:
                    test_button = st.form_submit_button("Test Connection")
                
                if connect_button or test_button:
                    if not all([account, user, password]):
                        st.error("Please fill in account, user, and password")
                    elif connect_button and not openai_key:
                        st.error("OpenAI API key is required for full functionality")
                    else:
                        connection_params = {
                            'account': account,
                            'user': user,  # Changed from 'username' to 'user'
                            'password': password,
                            'warehouse': warehouse if warehouse else None,
                            'database': database if database else None,
                            'schema': schema if schema else None,
                            'role': role if role else None
                        }
                        
                        if openai_key:
                            dashboard.settings.OPENAI_API_KEY = openai_key
                        
                        with st.spinner("Connecting to Snowflake and initializing agents..."):
                            success, message = dashboard.connect_to_snowflake(connection_params)
                            
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)

def render_profiling_tab(dashboard):
    """Render the profiling tab"""
    st.header("📊 Agentic Data Profiling")
    
    if not st.session_state.selected_table:
        st.info("👈 Please select a table from the sidebar")
        return
    
    st.subheader(f"Profiling: {st.session_state.selected_table}")
    
    # Check if we already have results for this table
    if st.session_state.selected_table in st.session_state.profiling_results:
        result = st.session_state.profiling_results[st.session_state.selected_table]
        
        if result.get('success'):
            # Display results
            st.success("✅ Profiling completed")
            
            # Show analysis summary
            analysis = result.get('analysis', {})
            
            if isinstance(analysis, str):
                st.markdown(analysis)
            elif isinstance(analysis, dict):
                # Display key metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Table", st.session_state.selected_table)
                with col2:
                    st.metric("Agent Used", "Yes" if result.get('agent_used') else "No")
                with col3:
                    if 'timestamp' in result:
                        st.metric("Completed", result['timestamp'][:19])
                
                # Display detailed analysis
                if 'analysis' in analysis:
                    st.markdown("### 📋 Analysis")
                    st.json(analysis['analysis'])
                    
                if 'recommendations' in analysis:
                    st.markdown("### 🎯 Recommendations")
                    for rec in analysis.get('recommendations', []):
                        st.info(rec)
        else:
            st.error(f"Profiling failed: {result.get('error', 'Unknown error')}")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Run Profiling", use_container_width=True):
            with st.spinner("Running agentic profiling..."):
                result, error = dashboard.run_profiling_workflow(st.session_state.selected_table)
                if error:
                    st.error(error)
                else:
                    st.success("Profiling completed!")
                    st.rerun()
    
    with col2:
        if st.button("📊 View Schema", use_container_width=True):
            if dashboard.orchestrator and dashboard.orchestrator.snowflake_manager:
                schema = dashboard.orchestrator.snowflake_manager.get_table_schema(st.session_state.selected_table)
                if not schema.empty:
                    st.dataframe(schema)
                else:
                    st.warning("Could not retrieve schema")
    
    with col3:
        if st.button("🧹 Clear Results", use_container_width=True):
            if st.session_state.selected_table in st.session_state.profiling_results:
                del st.session_state.profiling_results[st.session_state.selected_table]
            st.rerun()

def render_anomaly_tab(dashboard):
    """Render the anomaly detection tab"""
    st.header("🔍 Agentic Anomaly Detection")
    
    if not st.session_state.selected_table:
        st.info("👈 Please select a table from the sidebar")
        return
    
    st.subheader(f"Anomaly Detection: {st.session_state.selected_table}")
    
    # Check for existing results
    if st.session_state.selected_table in st.session_state.anomaly_results:
        result = st.session_state.anomaly_results[st.session_state.selected_table]
        
        if result.get('success'):
            st.success("✅ Anomaly detection completed")
            
            analysis = result.get('analysis', {})
            
            if isinstance(analysis, dict):
                # Display anomaly summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_anomalies = analysis.get('total_anomalies', 0)
                    st.metric("Total Anomalies", total_anomalies)
                with col2:
                    critical = analysis.get('critical_anomalies', 0)
                    st.metric("Critical", critical, delta_color="inverse")
                with col3:
                    affected = analysis.get('affected_columns', 0)
                    st.metric("Affected Columns", affected)
                with col4:
                    agent_used = result.get('agent_used', False)
                    st.metric("Agent Used", "Yes" if agent_used else "No")
                
                # Show anomaly types
                if 'anomaly_types' in analysis:
                    st.markdown("### 📊 Anomaly Types")
                    anomaly_types = analysis['anomaly_types']
                    
                    for anomaly_type, count in anomaly_types.items():
                        st.progress(min(count / max(sum(anomaly_types.values()), 1), 1.0), 
                                  text=f"{anomaly_type}: {count}")
                
                # Show recommendations
                if 'recommendations' in analysis:
                    st.markdown("### 🚨 Recommendations")
                    for rec in analysis['recommendations']:
                        st.warning(rec)
                        
        else:
            st.error(f"Anomaly detection failed: {result.get('error', 'Unknown error')}")
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Run Anomaly Detection", use_container_width=True, type="primary"):
            with st.spinner("Running anomaly detection..."):
                result, error = dashboard.run_anomaly_detection_workflow(st.session_state.selected_table)
                if error:
                    st.error(error)
                else:
                    st.success("Anomaly detection completed!")
                    st.rerun()
    
    with col2:
        if st.button("🧹 Clear Anomaly Results", use_container_width=True):
            if st.session_state.selected_table in st.session_state.anomaly_results:
                del st.session_state.anomaly_results[st.session_state.selected_table]
            st.rerun()

def render_sql_agent_tab(dashboard):
    """Render the SQL agent tab"""
    st.header("🤖 Natural Language SQL Agent")
    
    if not st.session_state.selected_table:
        st.info("👈 Please select a table from the sidebar first")
        return
    
    st.info(f"💡 Ask questions about data in table: **{st.session_state.selected_table}**")
    
    # Query input
    query = st.text_area(
        "Ask a question in natural language:",
        placeholder=f"E.g., 'Show me the top 10 customers by total orders'",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🚀 Generate & Execute SQL", type="primary", use_container_width=True):
            if not query:
                st.error("Please enter a question")
            else:
                with st.spinner("🤖 Thinking... Generating SQL and executing..."):
                    result, error = dashboard.run_sql_generation_workflow(query)
                    
                    if error:
                        st.error(f"Error: {error}")
                    elif result:
                        if result.get('success'):
                            st.success("✅ Query executed successfully!")
                            
                            # Display SQL
                            st.markdown("### 📝 Generated SQL")
                            st.code(result.get('generated_sql', 'No SQL generated'), language='sql')
                            
                            # Display results
                            if 'data' in result and not result['data'].empty:
                                st.markdown("### 📊 Results")
                                st.dataframe(result['data'])
                                
                                # Show summary
                                summary = result.get('result_summary', {})
                                st.metric("Rows", summary.get('rows', 0))
                                st.metric("Columns", summary.get('columns', 0))
                        else:
                            st.error(f"Query failed: {result.get('error', 'Unknown error')}")
    
    # Query history
    if st.session_state.query_history:
        st.markdown("### 📜 Query History")
        
        for i, history_item in enumerate(reversed(st.session_state.query_history[-5:]), 1):
            with st.expander(f"Query {i}: {history_item['query'][:50]}..."):
                st.markdown(f"**Timestamp:** {history_item['timestamp']}")
                st.markdown(f"**Natural Language:** {history_item['query']}")
                st.markdown("**Generated SQL:**")
                st.code(history_item['sql'], language='sql')
                
                if not history_item['data'].empty:
                    st.markdown("**Results:**")
                    st.dataframe(history_item['data'].head(10))

def render_dashboard(dashboard):
    """Render the main dashboard"""
    st.header("📈 Data Quality Dashboard")
    
    if not st.session_state.selected_table:
        st.info("👈 Please select a table from the sidebar")
        return
    
    # Overall status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        has_profiling = st.session_state.selected_table in st.session_state.profiling_results
        st.metric(
            "Profiling", 
            "✅ Completed" if has_profiling else "⏳ Pending",
            delta="Ready" if has_profiling else "Not run"
        )
    
    with col2:
        has_anomalies = st.session_state.selected_table in st.session_state.anomaly_results
        st.metric(
            "Anomaly Detection",
            "✅ Completed" if has_anomalies else "⏳ Pending",
            delta="Ready" if has_anomalies else "Not run"
        )
    
    with col3:
        query_count = len(st.session_state.query_history)
        st.metric("SQL Queries", query_count)
    
    with col4:
        table_name = st.session_state.selected_table
        if dashboard.orchestrator and dashboard.orchestrator.snowflake_manager:
            try:
                row_count = dashboard.orchestrator.snowflake_manager.get_row_count(table_name)
                st.metric("Total Rows", f"{row_count:,}")
            except:
                st.metric("Total Rows", "N/A")
    
    # Quick actions row
    st.markdown("---")
    st.subheader("🚀 Quick Actions")
    
    action_cols = st.columns(4)
    
    with action_cols[0]:
        if st.button("📊 Full Profile", use_container_width=True):
            with st.spinner("Running full profile..."):
                result, error = dashboard.run_profiling_workflow(st.session_state.selected_table)
                if error:
                    st.error(error)
                else:
                    st.success("Profiling completed!")
                    st.rerun()
    
    with action_cols[1]:
        if st.button("🔍 Detect All Anomalies", use_container_width=True):
            with st.spinner("Detecting anomalies..."):
                result, error = dashboard.run_anomaly_detection_workflow(st.session_state.selected_table)
                if error:
                    st.error(error)
                else:
                    st.success("Anomaly detection completed!")
                    st.rerun()
    
    with action_cols[2]:
        if st.button("📋 Generate Report", use_container_width=True):
            if dashboard.orchestrator:
                with st.spinner("Generating report..."):
                    try:
                        result = dashboard.orchestrator.run_sync(
                            dashboard.orchestrator.run_report_generation_workflow(st.session_state.selected_table)
                        )
                        if result and result.get('success'):
                            st.success("Report generated!")
                            st.markdown(result.get('report', 'No report content'))
                        else:
                            st.error(f"Report generation failed: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Report generation error: {str(e)}")
    
    with action_cols[3]:
        if st.button("🧹 Clear All Results", use_container_width=True, type="secondary"):
            keys_to_clear = ['profiling_results', 'anomaly_results', 'query_history']
            for key in keys_to_clear:
                if key in st.session_state:
                    if key.endswith('_results'):
                        st.session_state[key] = {}
                    elif key == 'query_history':
                        st.session_state[key] = []
            st.rerun()