const { chromium } = require('playwright');
(async () => {
  const job = JSON.parse(process.argv[2]);
  const b = await chromium.launch();
  const pg = await b.newPage({ viewport: { width: 900, height: 1200 }, deviceScaleFactor: 1 });
  await pg.goto('file://' + job.html);
  for (const id of job.ids) await pg.locator('#' + id).screenshot({ path: `${job.out}/${id}.png` });
  if (job.pdf) {
    await pg.emulateMedia({ media: 'screen' });
    await pg.pdf({ path: job.pdf, width: '900px', height: '1560px', printBackground: true, margin: { top: 0, bottom: 0, left: 0, right: 0 } });
  }
  await b.close();
})();
