const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs');
const ctx=vm.createContext({});vm.runInContext(fs.readFileSync('tools/football_event_review/static/forward_buffer.js','utf8'),ctx);
const policy=opts=>ctx.forwardBufferPolicy({ready:4,seeking:false,failed:false,hidden:false,current:100,bufferedEnd:110,next:120,...opts});
test('nearby next context requests native prebuffer, stops after target plus context',()=>{assert.equal(policy({}),'auto');assert.equal(policy({bufferedEnd:128}),'metadata')});
test('distant, backward and absent targets do not preload',()=>{for(const next of [null,NaN,50,131])assert.equal(policy({next}),'metadata')});
test('current playback and constrained networks take priority',()=>{for(const opts of [{ready:2},{seeking:true},{failed:true},{hidden:true},{saveData:true},{effectiveType:'3g'},{downlink:.5}])assert.equal(policy(opts),'metadata')});
test('controller never seeks or reloads; stalled and hidden player back off',()=>{
 const events={},docEvents={};let now=100;const player={currentTime:100,readyState:4,seeking:false,error:null,preload:'metadata',buffered:{length:1,start:()=>90,end:()=>110},addEventListener:(n,f)=>events[n]=f};
 const document={hidden:false,addEventListener:(n,f)=>docEvents[n]=f};
 const c=vm.createContext({navigator:{},document,Date:{now:()=>now}});vm.runInContext(fs.readFileSync('tools/football_event_review/static/forward_buffer.js','utf8'),c);
 c.installForwardBuffer(player,()=>120);assert.equal(player.preload,'auto');events.waiting();events.canplay();assert.equal(player.preload,'metadata');now+=30001;events.canplay();assert.equal(player.preload,'auto');document.hidden=true;docEvents.visibilitychange();assert.equal(player.preload,'metadata');assert.equal(player.currentTime,100);
});
