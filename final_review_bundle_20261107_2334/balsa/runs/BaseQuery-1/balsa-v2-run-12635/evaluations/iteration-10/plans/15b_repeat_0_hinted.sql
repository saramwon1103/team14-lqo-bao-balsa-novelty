/*+ HashJoin(ct mc at it1 mi k mk cn t)
 NestLoop(ct mc at it1 mi k mk cn)
 NestLoop(mc at it1 mi k mk cn)
 HashJoin(mc at it1 mi k mk)
 HashJoin(at it1 mi k mk)
 HashJoin(it1 mi k mk)
 HashJoin(mi k mk)
 NestLoop(k mk)
 SeqScan(ct)
 SeqScan(mc)
 SeqScan(at)
 IndexScan(it1)
 SeqScan(mi)
 IndexScan(k)
 IndexScan(mk)
 IndexScan(cn)
 SeqScan(t)
 Leading(((ct ((mc (at (it1 (mi (k mk))))) cn)) t)) */
SELECT MIN(mi.info) AS release_date,
       MIN(t.title) AS youtube_movie
FROM aka_title AS at,
     company_name AS cn,
     company_type AS ct,
     info_type AS it1,
     keyword AS k,
     movie_companies AS mc,
     movie_info AS mi,
     movie_keyword AS mk,
     title AS t
WHERE cn.country_code = '[us]'
  AND cn.name = 'YouTube'
  AND it1.info = 'release dates'
  AND mc.note LIKE '%(200%)%'
  AND mc.note LIKE '%(worldwide)%'
  AND mi.note LIKE '%internet%'
  AND mi.info LIKE 'USA:% 200%'
  AND t.production_year BETWEEN 2005 AND 2010
  AND t.id = at.movie_id
  AND t.id = mi.movie_id
  AND t.id = mk.movie_id
  AND t.id = mc.movie_id
  AND mk.movie_id = mi.movie_id
  AND mk.movie_id = mc.movie_id
  AND mk.movie_id = at.movie_id
  AND mi.movie_id = mc.movie_id
  AND mi.movie_id = at.movie_id
  AND mc.movie_id = at.movie_id
  AND k.id = mk.keyword_id
  AND it1.id = mi.info_type_id
  AND cn.id = mc.company_id
  AND ct.id = mc.company_type_id;

