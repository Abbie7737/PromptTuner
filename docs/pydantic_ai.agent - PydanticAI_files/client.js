// flarelytics client 8359587
if (!window.flarelytics_send) {
  window.fl_referer = document.referrer;
  window.flarelytics_send = (extra) => {
    const url = new URL(window.fl_url);
    const data = Object.assign(
      {
        w: window.innerWidth,
        h: window.innerHeight,
        o: url.host,
        p: url.pathname,
        a: url.hash,
        q: url.search,
        r: window.fl_referer,
        c: '8359587',
      },
      extra,
    );
    const filtered_data = Object.entries(data).filter(([, v]) => v !== '');
    const url_query = new URLSearchParams(Object.fromEntries(filtered_data));
    navigator.sendBeacon(window.location.origin + '/flarelytics/ping?' + url_query);
  };

  window.fl_page_view = () => {
    window.fl_page_start = Date.now();
    window.fl_url = window.location.href;
    window.flarelytics_send();
  };
  window.fl_page_view();

  window.flarelytics_event = (event, time) => {
    const extra = {
      e: event,
      t: ((time || Date.now()) - window.fl_page_start).toString(),
      sc: window.pageYOffset,
      mx: window.fl_mx,
      my: window.fl_my,
    };
    window.flarelytics_send(extra);
  };
  window.fl_mx = '';
  window.fl_my = '';
  addEventListener('mousemove', (event) => {
    window.fl_mx = event.clientX;
    window.fl_my = event.clientY;
  });

  window.fl_visible = true;
  addEventListener('pagehide', () => window.flarelytics_event(window.fl_visible ? 'leave' : 'leave_hidden'));

  window.fl_last_vis = Date.now();
  window.fl_check = setInterval(() => {
    if (document.visibilityState === 'visible') {
      window.fl_last_vis = Date.now();
      const { href } = window.location;
      if (href !== window.fl_url) {
        window.flarelytics_event('leave_pushstate');
        // only mark the page as visible again once we navigate away
        // this should mean we only get one page_hidden per page_view
        window.fl_visible = true;
        window.fl_referer = window.fl_url;
        window.fl_page_view();
      }
    } else if (window.fl_visible && Date.now() - window.fl_last_vis > 30_000) {
      window.fl_visible = false;
      window.flarelytics_event('page_hidden', window.fl_last_vis);
    }
  }, 1000);
}
