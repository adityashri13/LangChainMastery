from bs4 import BeautifulSoup

def clean_html(html_content):
    """Extracts and cleans text from the body of HTML content, ignoring script and style tags."""
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract the body content
    body_content = soup.body
    
    # Remove all script and style tags from the body content
    for script_or_style in body_content(['script', 'style']):
        script_or_style.decompose()  # Removes the tag and its content
    
    # Extract text from the cleaned body content
    text = body_content.get_text()
    
    # Optionally, clean the text further, e.g., removing extra spaces
    clean_text = ' '.join(text.split())
    
    return clean_text

# Example HTML content
html_content = """
<html>
<head><title>My Web Page</title></head>
<body>
    <p>This is a paragraph of text.</p>
    <script>alert('This should be removed');</script>
    <style>p { color: red; }</style>
    <div>Another <span>piece of text</span> inside the body.</div>
</body>
</html>
"""

# Clean the HTML content
clean_text = clean_html(html_content)

print(clean_text)
