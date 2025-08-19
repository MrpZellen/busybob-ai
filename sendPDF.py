import emails, base64, asyncio
import os, datetime, pymongo
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, select_autoescape

load_dotenv()
PASS = os.getenv('PRIVPASS')

class CompanyInstance:
    def __init__(self, name, date):
        self.name = name
        self.date = date

templateEnvironment = Environment(
    loader=FileSystemLoader('busybob/html'),
    autoescape=select_autoescape(['html'])
)
template = templateEnvironment.get_template('emailHeader.html')

senderEmail = 'synersyllo@benleonard.net'
def youveGotMail(recieverEmail: str, pdfStr: str, pdfName: str, companyInfo: CompanyInstance):
    message = emails.Message(html=template.render(name=companyInfo.name, date=companyInfo.date), subject='SynerSyllo Weekly Report')
    pdfData = base64.b64decode(pdfStr)
    message.render(name=companyInfo.name, date=companyInfo.date)
    message.attach(data=pdfData, filename=pdfName)
    response = message.send(to=recieverEmail, mail_from=senderEmail, smtp={'host': 'smtp.gmail.com', 'timeout': 5, 'port': 587, 'tls': True, 'user': 'Ben@benleonard.net', 'password': PASS})
    if response.status_code not in [250]:
        print('failed to send email')
        print(response)
        return 400
    else:
        print('email send success!')
        print(response)
        return 200
    
async def getB64(companyID: str):
    print('reached connectDB')
    client = pymongo.MongoClient('mongodb://localhost:2200/survey_data?directConnection=true')
    db = client.survey_data
    collection = db.pdf_store
    result = collection.find_one(filter={'companyID': companyID})
    return result

foundData = asyncio.run(getB64('COM-newtest'))
now = datetime.date.today()
companyTest = CompanyInstance('johnco', now.strftime('%d/%m/%y'))
youveGotMail('jshull@neumont.edu', foundData['pdfStore'][0]['PDFstring'], 'data.pdf', companyTest)