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

CONNECTION_STRING = os.getenv('MONGODB_URI')

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


        #output
        dateResultString = reportDate.strftime('%d-%m-%y')
        if (os.path.exists(f'BobPDF/{companyName}') == False):
            os.makedirs(f'BobPDF/{companyName}')
        
        result = pdf.output(dest='S').encode('latin1')
        b64pdf = base64.b64encode(result).decode('ascii')
        docToInsert = ColDoc('COM-newtest', b64pdf)
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
    client = pymongo.MongoClient('mongodb://localhost:2200/survey_data?directConnection=true')
    db = client.survey_data
    collection = db.pdf_store
    if collection.find_one(filter={'companyID': insertedDoc.companyID}):
        print('sending update')
        result = collection.update_one(filter={'companyID': insertedDoc.companyID}, update={ '$push': {'pdfStore': myDictRes['pdfStore'][0] }})
    else:
        print('sending new doc')
        result = collection.insert_one(myDictRes)
    if result:
        return True
    else:
        return False


