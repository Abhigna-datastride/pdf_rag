import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import openai
import re
from typing import Dict, Any, List, Optional

class ChartGenerator:
    def __init__(self, openai_client):
        self.openai_client = openai_client
        
    def detect_chart_request(self, user_query: str) -> bool:
        """Detect if the user is requesting a chart or visualization."""
        chart_keywords = [
            'chart', 'graph', 'plot', 'visualize', 'visualization', 'bar chart', 
            'line chart', 'pie chart', 'histogram', 'scatter plot', 'show me',
            'create a chart', 'make a graph', 'draw', 'display graphically',
            'trend', 'comparison chart', 'distribution'
        ]
        
        query_lower = user_query.lower()
        return any(keyword in query_lower for keyword in chart_keywords)
    
    def extract_data_for_chart(self, retrieved_chunks: List[str], user_query: str) -> Optional[Dict[str, Any]]:
        """Extract and structure data from document chunks for chart creation."""
        
        # Combine chunks into context
        context = "\n\n".join(retrieved_chunks)
        
        system_prompt = """You are a data extraction expert. Your job is to:

1. Analyze the provided document content and user query
2. Extract relevant numerical data that can be used to create charts
3. Structure the data in a JSON format suitable for visualization
4. Determine the most appropriate chart type based on the data and query

IMPORTANT RULES:
- Only extract actual numerical data from the documents
- If no suitable numerical data exists, return {"error": "No chartable data found"}
- Structure data as: {"chart_type": "bar/line/pie/scatter", "data": [{"label": "...", "value": ...}], "title": "...", "x_label": "...", "y_label": "..."}
- Supported chart types: bar, line, pie, scatter
- Ensure all values are numeric
- Keep labels concise and clear

Examples of good responses:
- {"chart_type": "bar", "data": [{"label": "Q1", "value": 100}, {"label": "Q2", "value": 150}], "title": "Quarterly Sales", "x_label": "Quarter", "y_label": "Sales ($)"}
- {"chart_type": "pie", "data": [{"label": "Product A", "value": 40}, {"label": "Product B", "value": 60}], "title": "Market Share", "x_label": "", "y_label": ""}
"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user", 
                        "content": f"""User Query: "{user_query}"

Document Content:
{context}

Extract numerical data from the document content that can be used to create a chart based on the user's query. Return the result as a JSON object."""
                    }
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                chart_data = json.loads(json_match.group())
                return chart_data
            else:
                return {"error": "Could not parse chart data"}
                
        except Exception as e:
            return {"error": f"Error extracting chart data: {str(e)}"}
    
    def create_chart(self, chart_data: Dict[str, Any]) -> Optional[go.Figure]:
        """Create a Plotly chart based on the extracted data."""
        
        if "error" in chart_data:
            st.error(f"Chart generation failed: {chart_data['error']}")
            return None
        
        try:
            chart_type = chart_data.get("chart_type", "bar")
            data = chart_data.get("data", [])
            title = chart_data.get("title", "Chart")
            x_label = chart_data.get("x_label", "")
            y_label = chart_data.get("y_label", "")
            
            if not data:
                st.error("No data available for chart creation")
                return None
            
            # Convert data to pandas DataFrame
            df = pd.DataFrame(data)
            
            # Create chart based on type
            if chart_type == "bar":
                fig = px.bar(
                    df, 
                    x="label", 
                    y="value", 
                    title=title,
                    labels={"label": x_label, "value": y_label}
                )
                fig.update_layout(
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    showlegend=False
                )
                
            elif chart_type == "line":
                fig = px.line(
                    df, 
                    x="label", 
                    y="value", 
                    title=title,
                    labels={"label": x_label, "value": y_label}
                )
                fig.update_layout(
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    showlegend=False
                )
                
            elif chart_type == "pie":
                fig = px.pie(
                    df, 
                    values="value", 
                    names="label", 
                    title=title
                )
                
            elif chart_type == "scatter":
                fig = px.scatter(
                    df, 
                    x="label", 
                    y="value", 
                    title=title,
                    labels={"label": x_label, "value": y_label}
                )
                fig.update_layout(
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    showlegend=False
                )
                
            else:
                # Default to bar chart
                fig = px.bar(df, x="label", y="value", title=title)
            
            # Customize appearance
            fig.update_layout(
                template="plotly_white",
                height=500,
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            return None
    
    def generate_chart_response(self, user_query: str, retrieved_chunks: List[str]) -> tuple[Optional[go.Figure], str]:
        """Generate both chart and textual response for chart-related queries."""
        
        # Extract data for chart
        chart_data = self.extract_data_for_chart(retrieved_chunks, user_query)
        
        # Create chart
        chart = self.create_chart(chart_data) if chart_data else None
        
        # Generate textual response
        context = "\n\n".join(retrieved_chunks)
        
        system_prompt = """You are an intelligent document assistant with chart generation capabilities. 

When the user requests a chart or visualization:
1. Provide a brief explanation of what data was found
2. Describe the chart that was created (or explain why a chart couldn't be created)
3. Give insights about the data trends, patterns, or key findings
4. Keep the response concise but informative

If a chart was successfully created, mention that it's displayed above/below your response.
If no chart could be created, explain why and offer alternative ways to present the information."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user", 
                        "content": f"""User Query: "{user_query}"

Document Content:
{context}

Chart Data Extracted: {json.dumps(chart_data) if chart_data else "None"}

Provide a response that explains the data and chart (if created)."""
                    }
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            text_response = response.choices[0].message.content
            
        except Exception as e:
            text_response = f"I found some data in your documents, but encountered an error generating the response: {str(e)}"
        
        return chart, text_response

# Additional utility functions for advanced chart types
def create_multi_series_chart(data: List[Dict], chart_type: str = "bar") -> Optional[go.Figure]:
    """Create charts with multiple data series."""
    try:
        df = pd.DataFrame(data)
        
        if chart_type == "bar":
            fig = px.bar(df, x="category", y="value", color="series", barmode="group")
        elif chart_type == "line":
            fig = px.line(df, x="category", y="value", color="series")
        else:
            return None
            
        fig.update_layout(template="plotly_white", height=500)
        return fig
        
    except Exception as e:
        st.error(f"Error creating multi-series chart: {str(e)}")
        return None

def create_time_series_chart(data: List[Dict]) -> Optional[go.Figure]:
    """Create time-based charts."""
    try:
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        
        fig = px.line(df, x="date", y="value", title="Time Series Analysis")
        fig.update_layout(template="plotly_white", height=500)
        return fig
        
    except Exception as e:
        st.error(f"Error creating time series chart: {str(e)}")
        return None