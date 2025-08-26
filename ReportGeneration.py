from fpdf import FPDF
from dotenv import load_dotenv
import os, pymongo, datetime, base64, asyncio


class PDFStore:
    def __init__(self, creationDate, PDFstring):
        self.PDFstring = PDFstring
        self.creationDate = creationDate

class ColDoc:
    def __init__(self, companyID, PDFstring):
        self.companyID = companyID
        self.pdfStore = PDFStore(datetime.datetime.today(), PDFstring)

load_dotenv()

CONNECTION_STRING = os.getenv('MONGOPDF_URI')

if not CONNECTION_STRING:
    raise ValueError("CONNECTION_STRING is missing from environment variables")


async def GenerateDocument(reportDate: datetime.date, companyName: str, bobResponse):
    pdf = FPDF()
    # starting content

    # page 1: Overall Report Review
    pdf.add_page()

    # TEST Content
    if(companyName == 'BOBTEST'):
        pdf.set_font('Courier', 'B', 40)
        pdf.cell(w=0, h=10, txt='Sup gang!', align='C')
        pdf.ln(h=20)
        pdf.set_font('Courier', '', 14)
        pdf.multi_cell(w=0, h=8, txt='     second line! heres some paragraph text! bla bla bla bla bla bla im a vampire! from the top ropes my prince!!!')
        # output
        pdf.output('BobPDF/BobTest.pdf', 'F')
    else:
        # HEADER content
        pdf.set_font('Courier', 'B', 40)
        pdf.multi_cell(w=0, h=20, txt=companyName, align='C')
        pdf.set_font('Courier', 'I', 20)
        headerstring = f"BusyBOB report: {reportDate}"
        pdf.multi_cell(w=0, h=10, txt=headerstring, align='C')

        from fpdf import FPDF
from dotenv import load_dotenv
import os, pymongo, datetime, base64, asyncio, json

class PDFStore:
    def __init__(self, creationDate, PDFstring):
        self.PDFstring = PDFstring
        self.creationDate = creationDate

class ColDoc:
    def __init__(self, companyID, PDFstring):
        self.companyID = companyID
        self.pdfStore = PDFStore(datetime.datetime.today(), PDFstring)

load_dotenv()
CONNECTION_STRING = os.getenv('MONGODB_URI')
if not CONNECTION_STRING:
    raise ValueError("CONNECTION_STRING is missing from environment variables")

async def GenerateDocument(reportDate: datetime.date, companyName: str, bobResponse, companyID):
    pdf = FPDF()
    print('we here', companyID)
    # Parse the JSON response
    try:
        if isinstance(bobResponse, str):
            response_data = json.loads(bobResponse)
        else:
            response_data = bobResponse
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        response_data = {
            "generalDescription": "Survey analysis completed",
            "toneNotes": "professional",
            "healthRating": 75,
            "dataTrends": {
                "sentimentTrend": "Analysis in progress",
                "tagTrends": "Feedback categorized"
            },
            "finalThoughts": "Report generated successfully"
        }
    
    # starting content
    # page 1: Overall Report Review
    pdf.add_page()
    
    # TEST Content
    if(companyName == 'BOBTEST'):
        pdf.set_font('Courier', 'B', 40)
        pdf.cell(w=0, h=10, txt='Sup gang!', align='C')
        pdf.ln(h=20)
        pdf.set_font('Courier', '', 14)
        pdf.multi_cell(w=0, h=8, txt=' second line! heres some paragraph text! bla bla bla bla bla bla im a vampire! from the top ropes my prince!!!')
        # output
        pdf.output('BobPDF/BobTest.pdf', 'F')
    else:
        # HEADER content
        pdf.set_font('Courier', 'B', 40)
        pdf.multi_cell(w=0, h=20, txt=companyName, align='C')
        pdf.set_font('Courier', 'I', 20)
        headerstring = f"BusyBOB report: {reportDate}"
        pdf.multi_cell(w=0, h=10, txt=headerstring, align='C')
        
        # Add some spacing
        pdf.ln(h=15)
        
        # HEALTH RATING SECTION
        pdf.set_font('Courier', 'B', 16)
        pdf.cell(w=0, h=10, txt=f"Health Rating: {response_data.get('healthRating', 'N/A')}/100", ln=True)
        pdf.ln(h=5)
        
        # GENERAL DESCRIPTION SECTION
        pdf.set_font('Courier', 'B', 14)
        pdf.cell(w=0, h=8, txt="Executive Summary:", ln=True)
        pdf.set_font('Courier', '', 12)
        description = response_data.get('generalDescription', 'No description available')
        pdf.multi_cell(w=0, h=6, txt=description)
        pdf.ln(h=8)
        
        # TONE NOTES SECTION
        pdf.set_font('Courier', 'B', 14)
        pdf.cell(w=0, h=8, txt="Analysis Style:", ln=True)
        pdf.set_font('Courier', '', 12)
        tone_notes = response_data.get('toneNotes', 'Standard analysis')
        pdf.multi_cell(w=0, h=6, txt=f"Report tone: {tone_notes}")
        pdf.ln(h=8)
        
        # DATA TRENDS SECTION
        pdf.set_font('Courier', 'B', 14)
        pdf.cell(w=0, h=8, txt="Key Findings:", ln=True)
        pdf.set_font('Courier', '', 12)
        
        data_trends = response_data.get('dataTrends', {})
        
        # Sentiment Trend
        sentiment_trend = data_trends.get('sentimentTrend', 'No sentiment data available')
        pdf.multi_cell(w=0, h=6, txt=f"Sentiment Analysis: {sentiment_trend}")
        pdf.ln(h=3)
        
        # Tag Trends
        tag_trends = data_trends.get('tagTrends', 'No tag data available')
        pdf.multi_cell(w=0, h=6, txt=f"Category Analysis: {tag_trends}")
        pdf.ln(h=8)
        
        # FINAL THOUGHTS SECTION
        pdf.set_font('Courier', 'B', 14)
        pdf.cell(w=0, h=8, txt="Recommendations:", ln=True)
        pdf.set_font('Courier', '', 12)
        final_thoughts = response_data.get('finalThoughts', 'Analysis complete')
        pdf.multi_cell(w=0, h=6, txt=final_thoughts)
        
        #output
        dateResultString = reportDate.strftime('%d-%m-%y')
        if (os.path.exists(f'BobPDF/{companyName}') == False):
            os.makedirs(f'BobPDF/{companyName}')
        print('here past pdf stuff')
        result = pdf.output(dest='S').encode('latin1')
        b64pdf = base64.b64encode(result).decode('ascii')
        docToInsert = ColDoc(companyID, b64pdf)
        print('inserting doc')
        await insertB64(docToInsert)
        return b64pdf

        
async def insertB64(insertedDoc: ColDoc):
    myDictRes = {
        "companyID": insertedDoc.companyID,
        "pdfStore": [
            {
                "creationDate": insertedDoc.pdfStore.creationDate,
                "PDFstring": insertedDoc.pdfStore.PDFstring
            }
        ]
    }
    print('reached connectDB')
    client = pymongo.MongoClient(CONNECTION_STRING)
    db = client.survey_data
    collection = db.pdf_store
    if collection.find_one(filter={'companyID': insertedDoc.companyID}):
        print('sending update')
        result = collection.update_one(filter={'companyID': insertedDoc.companyID}, update={ '$push': {'pdfStore': myDictRes['pdfStore'][0] }})
        print(result)
    else:
        print('sending new doc')
        result = collection.insert_one(myDictRes)
        print('resultOfInsert', result)
    if result is not None:
        return result
    else:
        return False


