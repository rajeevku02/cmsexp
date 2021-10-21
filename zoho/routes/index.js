var express = require('express');
var router = express.Router();

const client_id = '1000.R7O7HMUWY310OC01A7U19S3WBZUS2D'
const client_secret = '7a8004e085d06ea9fe78c22a37217165500e7addcd'
const scope = 'ZohoMail.messages.READ,ZohoMail.messages.CREATE'
const redirect_uri = 'http://localhost:4000';
const auth_url = `https://accounts.zoho.com/oauth/v2/auth?scope=${scope}&client_id=${client_id}&response_type=code&access_type=online&redirect_uri=${redirect_uri}`

console.log(auth_url)

/* GET home page. */
router.get('/', function(req, res, next) {
  //res.render('index', { title: 'Express' });
  res.redirect(auth_url);
});

module.exports = router;
