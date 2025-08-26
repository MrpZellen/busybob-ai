import emails, base64, asyncio
import os, datetime, pymongo
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, select_autoescape

load_dotenv()
PASS = os.getenv('PRIVPASS')
CONNECTION_STRING = os.getenv('MONGOPDF_URI')

class CompanyInstance:
    def __init__(self, name, date):
        self.name = name
        self.date = date

templateEnvironment = Environment(
    loader=FileSystemLoader('html'),
    autoescape=select_autoescape(['html'])
)
template = templateEnvironment.get_template('emailHeader.html')

senderEmail = 'synersyllo@benleonard.net'
def youveGotMail(recieverEmail: str, pdfStr: str, pdfName: str, companyInfo: CompanyInstance):
    print("we've got mail", pdfStr)
    message = emails.Message(html=template.render(name=companyInfo.name, date=companyInfo.date), subject='SynerSyllo Weekly Report')
    pdfData = base64.b64decode(pdfStr)
    print('no beef here')
    message.render(name=companyInfo.name, date=companyInfo.date)
    message.attach(data=pdfData, filename=pdfName)
    with open("assets/icon-png/longo.png", "rb") as f:
        message.attach(data=f.read(), filename="longo.png", cid="<longo>")

    with open("assets/icon-png/Icon.png", "rb") as f:
        message.attach(data=f.read(), filename="Icon.png", cid="<busybob>")
    
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
    client = pymongo.MongoClient(CONNECTION_STRING)
    db = client.survey_data
    collection = db.pdf_store
    result = collection.find_one(filter={'companyID': companyID})
    return result
