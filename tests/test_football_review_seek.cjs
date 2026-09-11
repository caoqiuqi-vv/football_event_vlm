const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs'),path=require('node:path');
const code=fs.readFileSync(path.join(__dirname,'../tools/football_event_review/static/reliable_review.js'),'utf8').split('// Reuse the demuxer/index')[1];
function fixture(ready=1){
 const listeners=new Map();const player={readyState:ready,error:null,src:ready?'/media/v0':'',loads:0,plays:0,currentTime:0,
  getAttribute(){return this.src},load(){this.loads++},play(){this.plays++;return Promise.resolve()},
  addEventListener(name,fn){listeners.set(fn,name)},removeEventListener(name,fn){listeners.delete(fn)}};
 const state={video:{duration_sec:1000},currentVideoId:'v0',pendingAbsoluteSeek:null};
 const context=vm.createContext({state,$:selector=>selector==='#player'?player:null,mediaUrlAt:t=>'/media/v0#t='+t,formatTime:String,toast:()=>{}});
 vm.runInContext('// Reuse the demuxer/index'+code,context);
 return {player,state,listeners,seek:(t,play=true)=>vm.runInContext(`seekAbsolute(${t},${play})`,context)};
}
test('unbuffered distant seek reuses metadata and source',()=>{const f=fixture();f.seek(800);assert.equal(f.player.currentTime,800);assert.equal(f.player.loads,0);assert.equal(f.player.src,'/media/v0');assert.equal(f.player.plays,1)});
test('repeated seek during initial loading starts only one transfer and applies latest target',()=>{const f=fixture(0);f.seek(100);f.seek(200);assert.equal(f.player.loads,1);assert.equal(f.listeners.size,1);for(const fn of f.listeners.keys())fn();assert.equal(f.player.currentTime,200)});
test('stale metadata callback cannot seek a different video',()=>{const f=fixture(0);f.seek(100);f.state.currentVideoId='v2';for(const fn of f.listeners.keys())fn();assert.equal(f.player.currentTime,0)});
test('failed resource reloads once and resumes at requested time',()=>{const f=fixture();f.player.error={code:2};f.seek(400);assert.equal(f.player.loads,1);for(const fn of f.listeners.keys())fn();assert.equal(f.player.currentTime,400)});
