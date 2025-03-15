const express = require('express');
const jsforce = require('jsforce');
const dotenv = require('dotenv');
dotenv.config();
const app = express();

const PORT = process.env.PORT || 3000;

const SF_USERNAME = process.env.SF_USERNAME;
const SF_PASSWORD = process.env.SF_PASSWORD;
//const SF_SECURITY_TOKEN = process.env.SF_SECURITY_TOKEN;
const SF_LOGIN_URL = process.env.SF_LOGIN_URL;

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

app.get('/summary/:accountId', async (req, res) => {
    const accountId = req.params.accountId;

    try {
        const conn = new jsforce.Connection({ loginUrl: SF_LOGIN_URL });
        await conn.login(SF_USERNAME, SF_PASSWORD);
       
        let queryStr = `SELECT Description,ActivityDate FROM Task WHERE ActivityDate!=null and AccountId = '${accountId}' AND ActivityDate >= LAST_N_YEARS:4 ORDER BY ActivityDate DESC`;
        let records = [];
        let result = await conn.query(queryStr);

        records = records.concat(result.records);

        // Fetch additional records using queryMore
        while (!result.done) {
            result = await conn.queryMore(result.nextRecordsUrl);
            records = records.concat(result.records);
        }
        console.log(records.length);
        // Process the records into a summary
        const summary = records.map(r => `• ${r.ActivityDate} on ${r.Description}`).join("\n");

        res.json({ accountId, summary });

    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Failed to fetch data' });
    }
});

