'use strict';
/* ------------------------------------------------------------------ utils */
const $ = id => document.getElementById(id);
const fmt = n => (n ?? 0).toLocaleString('en-US');
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

function dec(b64, T){
  const s = atob(b64), u = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
  return new T(u.buffer);
}
function toRGB(s){
  s = s.trim();
  if (s[0] === '#'){
    if (s.length === 4) s = '#'+s[1]+s[1]+s[2]+s[2]+s[3]+s[3];
    return [parseInt(s.slice(1,3),16), parseInt(s.slice(3,5),16), parseInt(s.slice(5,7),16)];
  }
  const m = s.match(/[\d.]+/g); return [+m[0], +m[1], +m[2]];
}
function ramp256(stops){
  const out = new Array(256), n = stops.length-1, rgb = stops.map(toRGB);
  for (let i = 0; i < 256; i++){
    const t = i/255*n, k = Math.min(n-1, t|0), f = t-k, a = rgb[k], b = rgb[k+1];
    out[i] = `rgb(${Math.round(a[0]+(b[0]-a[0])*f)},${Math.round(a[1]+(b[1]-a[1])*f)},${Math.round(a[2]+(b[2]-a[2])*f)})`;
  }
  return out;
}
const NSH = 14;
function shades(cols){
  const rgb = cols.map(toRGB), out = new Array(cols.length*NSH);
  for (let c = 0; c < cols.length; c++) for (let q = 0; q < NSH; q++){
    const f = .42 + .70*(q/(NSH-1));
    out[c*NSH+q] = `rgb(${Math.min(255,rgb[c][0]*f|0)},${Math.min(255,rgb[c][1]*f|0)},${Math.min(255,rgb[c][2]*f|0)})`;
  }
  return out;
}
const SPECTRAL = ramp256(['#7A1E8C','#2A2FB4','#0B7FD6','#0FC3B4','#57D63A','#CFE01C','#FFA51C','#E32A1C']);
const CLASSC = ['#9A8C7A','#7A8899','#C05A5A','#E0A33E','#5D9B6B','#6F5FA8','#E0457B','#A0A6AD'];
const CLASSN = ['ground','road','building','pole','vegetation','car','pedestrian','other'];
const DETC = ['#8E9A86','#3E6E9C','#6F5FA8','#E0457B','#E0A33E','#B5654A','#CBD1D8'];
const DETN = ['ground (geometric)','examined, rejected','car','pedestrian','cyclist','never clustered','no cell here'];
let SP = null, CL = null, TV = null, DT = null, MT = null;
const pal = () => { if (!SP){ SP = shades(SPECTRAL); CL = shades(CLASSC); TV = shades([css('--ink3'), css('--no'), css('--ok')]); DT = shades(DETC); MT = shades([css('--ink3'), css('--ok'), css('--bad')]); } };

// ... visualization code retained unchanged from the Stage 6C build ...
