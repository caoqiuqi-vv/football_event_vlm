/* Run against the disposable server created by run_reliable_browser_test.py. */
const assert = require('node:assert/strict');
const {chromium} = require(process.env.REVIEW_PLAYWRIGHT_MODULE || 'playwright');
(async () => {
  const browser = await chromium.launch({headless: true, args: ['--no-sandbox']});
  try {
    const page = await browser.newPage();
    await page.addInitScript(() => { window.confirmWithTeam = () => {
      if (state.selectedSegmentLabels.has('shot')) state.attributionByLabel.shot = {...state.attributionByLabel.shot, event_team:'teamA'};
      return saveMultiLabelSegment();
    }; });
    const errors=[]; page.on('pageerror',error=>{errors.push(error.message);});
    page.on('dialog', dialog => dialog.accept());
    await page.goto(process.argv[2]+'/u/u0');
    await page.locator('input[name=password]').fill('test-only');
    await Promise.all([page.waitForURL('**/'),page.locator('button[type=submit]').click()]);
    await page.waitForLoadState('domcontentloaded');
    await page.waitForFunction(()=>typeof outboxReady!=='undefined' && outboxReady && state.video?.video_id==='v0');
    await page.evaluate(()=>playEventContext(currentEvent()));
    await page.waitForTimeout(1000);
    await page.waitForFunction(()=>document.querySelector('#player').readyState>=2);
    // Drop a write, reload, and recover from persisted storage with the same id.
    await page.route('**/api/events/*/segment-decision',route=>route.abort());
    await page.evaluate(()=>confirmWithTeam());
    await page.waitForFunction(()=>reviewOutbox.list()[0]?.status==='retry');
    const operation=await page.evaluate(()=>reviewOutbox.list()[0].id);
    await page.reload();
    await page.waitForFunction(()=>typeof outboxReady!=='undefined' && outboxReady && state.video);
    assert.equal(await page.evaluate(()=>reviewOutbox.list()[0].id),operation);
    await page.unroute('**/api/events/*/segment-decision');
    await page.evaluate(()=>reviewOutbox.retry(reviewOutbox.list()[0].id));
    await page.waitForFunction(()=>reviewOutbox.list().length===0);
    assert.equal(await page.evaluate(async()=> (await api('/api/videos/v0')).events.find(e=>e.id==='v0-e0').review.revision),1);
    // Commit a second event, lose the response, retry: one revision only.
    let dropped=false;
    await page.route('**/api/events/*/segment-decision',async route=>{
      if(!dropped){dropped=true;await route.fetch();await route.abort();}else await route.continue();
    });
    await page.evaluate(()=>{selectEvent('v0-e1',true,false);return confirmWithTeam();});
    await page.waitForFunction(()=>reviewOutbox.list().length===0);
    assert.equal(await page.evaluate(async()=> (await api('/api/videos/v0')).events.find(e=>e.id==='v0-e1').review.revision),1);
    await page.unroute('**/api/events/*/segment-decision');
    // Force an uncancellable late response for A after B has already loaded.
    const result=await page.evaluate(async()=>{
      const original=api;
      api=async(path,options)=>{const data=await original(path);if(path==='/api/videos/v0')await new Promise(r=>setTimeout(r,250));return data;};
      try {await Promise.all([loadVideo('v0'),loadVideo('v2')]);return [state.currentVideoId,state.video.video_id,state.mediaBaseUrl];}
      finally{api=original;}
    });
    assert.equal(result[0],'v2');assert.equal(result[1],'v2');
    assert.equal(await page.locator('#videoSelect').inputValue(),'v2');assert.match(result[2],/^\/media\/v2/);
    // Storage failure cannot mark an event reviewed or advance selection.
    const before=await page.evaluate(()=>[currentEvent().id,currentEvent().review.status]);
    await page.evaluate(async()=>{
      const original=Storage.prototype.setItem;
      Storage.prototype.setItem=function(key,value){if(key.startsWith('football-review-v40:'))throw new Error('quota injected');return original.call(this,key,value);};
      try{await confirmWithTeam();}finally{Storage.prototype.setItem=original;}
    });
    assert.deepEqual(await page.evaluate(()=>[currentEvent().id,currentEvent().review.status]),before);
    // Preserve the existing GT deletion guard.
    await page.evaluate(()=>{currentEvent().matching_gt_times=[10];state.selectedSegmentLabels.clear();reviewedGtAnchors.clear();});
    await page.evaluate(()=>confirmWithTeam());
    assert.equal(await page.evaluate(()=>reviewOutbox.list().length),0);
    await page.evaluate(()=>{delete currentEvent().matching_gt_times;state.selectedSegmentLabels=new Set(['shot']);});
    // A stale tab keeps its draft, reloads current revisions, then explicitly resubmits.
    await page.evaluate(async()=>{const e=currentEvent();await api(`/api/events/${e.id}/segment-decision`,{method:'POST',body:JSON.stringify({operation_id:newOperationId(),selected_labels:['shot'],attribution_by_label:{shot:{event_team:'teamA'}},active_label:'shot',compact_response:true,expected_revisions:{[e.id]:e.review.revision}})});});
    await page.evaluate(()=>confirmWithTeam());
    await page.waitForFunction(()=>reviewOutbox.list()[0]?.status==='conflict');
    await page.evaluate(()=>editQueuedOperation(reviewOutbox.list()[0]));
    await page.evaluate(()=>confirmWithTeam());
    await page.waitForFunction(()=>reviewOutbox.list().length===0);
    assert.equal(await page.evaluate(async()=> (await api('/api/videos/v2')).events.find(e=>e.id==='v2-e0').review.revision),2);
    // Event-specific fields: real control clicks plus actual server persistence.
    await page.evaluate(()=>loadVideo('v0'));
    for (const [offset,label,detail] of [[5,'save',null],[6,'shot',null],[7,'set_piece','corner'],[8,'set_piece','free_kick'],[9,'set_piece','penalty'],[10,'throw_in',null]]) {
      await page.evaluate(({offset,label,detail})=>{
        selectEvent(`v0-e${offset}`,true,false);
        state.selectedSegmentLabels=new Set([label]); state.secondaryLabels=new Set(detail?[detail]:[]);
        state.attributionByLabel={}; renderTeamAttribution();
      },{offset,label,detail});
      const card=page.locator(`#attributionCards .label-${label}`);
      if(label==='throw_in') assert.equal(await card.count(),0);
      else {
        assert.equal(await card.locator('[data-side]').count(),0);
        assert.equal(await card.locator('[data-team]').count(),label==='save'?0:2);
        assert.equal(await card.locator('[data-field-side]').count(),label==='save'?2:0);
        await page.evaluate(()=>saveMultiLabelSegment());
        assert.equal(await page.evaluate(()=>reviewOutbox.list().length),0,'missing required field must not enqueue');
        await card.locator(label==='save'?'[data-field-side="right"]':'[data-team="teamB"]').click();
      }
      await page.evaluate(()=>saveMultiLabelSegment());
      await page.waitForFunction(()=>reviewOutbox.list().length===0);
      const review=await page.evaluate(async({offset,label})=>{
        const video=await api('/api/videos/v0');return video.events.find(e=>e.segment_id===`v0-s${offset}` && e.label===label).review;
      },{offset,label});
      assert.ok(['accepted','modified'].includes(review.status));
      assert.equal(review.event_team,['save','throw_in'].includes(label)?'unknown':'teamB');
      assert.equal(review.field_side,label==='save'?'right':'unknown');
    }
    assert.deepEqual(errors,[]);
    const panelBox=await page.locator('#saveQueuePanel').boundingBox();
    const headerBox=await page.locator('header.topbar').boundingBox();
    assert.ok(panelBox.y>=headerBox.y && panelBox.y+panelBox.height<=headerBox.y+headerBox.height,'save status must fit inside header');
    if(process.env.REVIEW_SCREENSHOT) await page.screenshot({path:process.env.REVIEW_SCREENSHOT});
    console.log(JSON.stringify({event_specific_attribution:true,login:true,playable_media:true,offline_reload_recovered:true,lost_response_exactly_once:true,reverse_video_responses:true,quota_fail_closed:true,gt_deletion_guard:true,conflict_draft_resubmitted:true,test_video_codec:'VP9 in MP4 (test Chromium has no H.264 decoder)',page_errors:errors},null,2));
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
