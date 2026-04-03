import sys
from lxml import etree

if len(sys.argv) < 2:
    print("Missing: <.xml>")
    print(f"e.g.\npython {sys.argv[0]} biorobots.xml")
    sys.exit()

model_name = sys.argv[1]

schema_file = "schema.xsd"
print("Validating against: ",schema_file)
# Load the schema
with open(schema_file, "rb") as f:
    schema_doc = etree.parse(f)
schema = etree.XMLSchema(schema_doc)

# Parse and validate the XML
with open(model_name, "rb") as f:
    xml_doc = etree.parse(f)

if schema.validate(xml_doc):
    print("XML is valid")
else:
    print("XML is invalid:")
    for error in schema.error_log:
        print(f"  Line {error.line}: {error.message}")
