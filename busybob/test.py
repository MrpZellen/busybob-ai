from jinja2 import Template

template = Template("Hello {{ name }}, date is {{ date }}")
print(template.render(name="John", date="2025-08-11"))